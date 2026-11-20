"""GST invoices as PDFs, built from a quote (ticket C8).

Every amount is whole paise and comes straight from the quote: the invoice never recomputes a price. The quote's
integrity is checked first (`verify_quote`), and the invoice carries the quote's `terms_hash`, so an invoice can always
be matched to the exact terms the customer agreed to.

Until the company is registered and has a GSTIN, invoices are marked "TEST INVOICE: not a tax invoice". The tax lines
show one GST line at the quote's rate. Splitting it into CGST and SGST, or IGST, depends on the place of supply and
needs an accountant; the SAC code is also an accountant's decision, so neither is guessed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from fpdf import FPDF

from ekido_core.quotes import GST_BPS, QuoteError, rupees, verify_quote

_GSTIN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class InvoiceError(ValueError):
    """The invoice can't be made from what was given. The message is safe to show a customer."""


def gstin_is_valid(gstin: str) -> bool:
    """Format and check character of a 15-character GSTIN."""
    if not isinstance(gstin, str) or not _GSTIN.match(gstin):
        return False
    total = 0
    for index, char in enumerate(gstin[:14]):
        value = _GST_CHARS.index(char) * (1 if index % 2 == 0 else 2)
        total += value // 36 + value % 36
    return gstin[14] == _GST_CHARS[(36 - total % 36) % 36]


@dataclass(frozen=True)
class Party:
    name: str
    address_lines: tuple[str, ...] = ()
    gstin: str | None = None  # a buyer without a GSTIN is fine; a given one must be valid

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvoiceError("a name is needed on the invoice")
        if self.gstin is not None and not gstin_is_valid(self.gstin):
            raise InvoiceError("the GSTIN is not valid; check all 15 characters")


def financial_year(when: datetime) -> str:
    """Indian financial year, April to March: 2026-11-19 is '2026-27' and 2027-03-31 is also '2026-27'."""
    start = when.year if when.month >= 4 else when.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def invoice_number(sequence: int, issued_at: datetime) -> str:
    """EKI/2026-27/000123. The sequence is unbroken within a financial year; the caller keeps the counter."""
    if type(sequence) is not int or sequence < 1:
        raise InvoiceError("the invoice sequence must be a whole number from 1")
    return f"EKI/{financial_year(issued_at)}/{sequence:06d}"


def build_invoice(quote: dict, *, number: str, issued_at: datetime, buyer: Party, seller: Party, test_mode: bool = True) -> dict:
    """The invoice's data. Every amount is copied from the quote; the checks only confirm they add up."""
    try:
        verify_quote(quote, now=datetime.fromisoformat(quote["created_at"]))  # integrity only: a paid quote may have expired
    except (QuoteError, KeyError, ValueError) as exc:
        raise InvoiceError("the quote can't be verified, so no invoice was made") from exc
    if quote.get("currency") != "INR":
        raise InvoiceError("only rupee quotes can be invoiced")
    taxable, tax, total = quote["training_paise"], quote["gst_paise"], quote["total_paise"]
    if total != taxable + tax or quote["compute_paise"] + quote["distillation_paise"] != taxable:
        raise InvoiceError("the quote's amounts don't add up, so no invoice was made")
    lines = [{"description": f"Model training: {quote['recipe_id']} on {quote['base_model']}", "taxable_paise": quote["compute_paise"]}]
    if quote["distillation_paise"]:
        lines.append({"description": "Generated training examples (distillation)", "taxable_paise": quote["distillation_paise"]})
    return {
        "number": number,
        "issued_at": issued_at.isoformat(timespec="seconds"),
        "test_mode": bool(test_mode),
        "seller": {"name": seller.name, "address_lines": list(seller.address_lines), "gstin": seller.gstin},
        "buyer": {"name": buyer.name, "address_lines": list(buyer.address_lines), "gstin": buyer.gstin},
        "lines": lines,
        "taxable_paise": taxable,
        "gst_rate_bps": GST_BPS,
        "gst_paise": tax,
        "total_paise": total,
        "terms_hash": quote["terms_hash"],
    }


def _text(value: str) -> str:
    """The built-in PDF fonts cover Latin-1 only; anything else becomes '?' rather than failing the invoice."""
    return value.encode("latin-1", "replace").decode("latin-1")


def _money(paise: int) -> str:
    return rupees(paise).replace("₹", "Rs ")  # the built-in fonts have no rupee sign


def render_pdf(invoice: dict) -> bytes:
    """A one-page A4 PDF. Not compressed, so it stays small and its text can be checked."""
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_compression(False)
    pdf.set_creation_date(datetime.fromisoformat(invoice["issued_at"]))
    pdf.set_title(_text(f"Invoice {invoice['number']}"))
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    left, width = 15, 180

    def row(text: str, size: int = 10, bold: bool = False, align: str = "L", h: float = 6) -> None:
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_x(left)
        pdf.cell(width, h, _text(text), align=align, new_x="LMARGIN", new_y="NEXT")

    row("Ekido", 20, True, h=10)
    row("TAX INVOICE" if not invoice["test_mode"] else "TEST INVOICE: not a tax invoice", 12, True, h=8)
    row(f"Invoice no: {invoice['number']}")
    row(f"Date: {invoice['issued_at'][:10]}")
    pdf.ln(3)
    for title, party in (("From", invoice["seller"]), ("Bill to", invoice["buyer"])):
        row(f"{title}: {party['name']}", bold=True)
        for line in party["address_lines"]:
            row(line)
        row(f"GSTIN: {party['gstin'] or 'not given'}")
        pdf.ln(2)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(130, 7, "Description", border="B")
    pdf.cell(50, 7, "Taxable value", border="B", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in invoice["lines"]:
        pdf.cell(130, 7, _text(line["description"]))
        pdf.cell(50, 7, _money(line["taxable_paise"]), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    rate = f"{invoice['gst_rate_bps'] / 100:g}%"
    for label, value, bold in (("Taxable value", invoice["taxable_paise"], False), (f"GST ({rate})", invoice["gst_paise"], False), ("Total", invoice["total_paise"], True)):
        pdf.set_font("Helvetica", "B" if bold else "", 11 if bold else 10)
        pdf.cell(130, 7, label, align="R")
        pdf.cell(50, 7, _money(value), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    row(f"Quote terms hash: {invoice['terms_hash']}", 8)
    row("Amounts are in Indian rupees. GST is shown separately.", 8)
    if invoice["test_mode"]:
        row("Test period: no real money was charged.", 8)
    return bytes(pdf.output())
