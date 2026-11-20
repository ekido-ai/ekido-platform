import copy
import unittest
from datetime import datetime, timezone

from ekido_core.quotes import GST_BPS, gst, make_quote, rupees

from ekido_platform import invoices
from ekido_platform.invoices import InvoiceError, Party, build_invoice, financial_year, gstin_is_valid, invoice_number, render_pdf

NOW = datetime(2026, 11, 19, 9, 30, tzinfo=timezone.utc)
GOOD_GSTIN = "27AAAPA1234A1Z5"  # replaced below by a computed valid one
SELLER = Party("Ekido (test)", ("Mumbai, India",), None)
BUYER = Party("Sharma Traders", ("Shop 4, Lake View Colony", "Pune 411001"), None)


def valid_gstin() -> str:
    body = "27AAAPA1234A1Z"
    total = 0
    chars = invoices._GST_CHARS
    for index, char in enumerate(body):
        value = chars.index(char) * (1 if index % 2 == 0 else 2)
        total += value // 36 + value % 36
    return body + chars[(36 - total % 36) % 36]


def quote(rows=200, distill=False, model="Qwen/Qwen2.5-0.5B-Instruct", recipe="ticket-sorter-hinglish"):
    return make_quote(recipe_id=recipe, base_model=model, train_rows=rows, test_rows=40, dataset_sha256="a" * 64, distill=distill, now=NOW)


class GstMaths(unittest.TestCase):
    def test_gst_is_eighteen_percent_rounded_to_the_nearest_paisa(self):
        self.assertEqual(GST_BPS, 1800)
        for taxable, expected in ((0, 0), (1, 0), (3, 1), (5, 1), (6, 1), (2500, 450), (2778, 500), (2779, 500), (2780, 500), (999_999, 180_000), (100_000, 18_000)):
            with self.subTest(taxable=taxable):
                self.assertEqual(gst(taxable), expected)

    def test_total_is_taxable_plus_gst_for_many_quotes_and_the_invoice_copies_it(self):
        for rows in (20, 50, 137, 500, 2000, 9999):
            for distill in (False, True):
                q = quote(rows, distill)
                inv = build_invoice(q, number="EKI/2026-27/000001", issued_at=NOW, buyer=BUYER, seller=SELLER)
                with self.subTest(rows=rows, distill=distill):
                    self.assertEqual(inv["taxable_paise"] + inv["gst_paise"], inv["total_paise"])
                    self.assertEqual(sum(line["taxable_paise"] for line in inv["lines"]), inv["taxable_paise"])
                    self.assertEqual((inv["taxable_paise"], inv["gst_paise"], inv["total_paise"]), (q["training_paise"], q["gst_paise"], q["total_paise"]))
                    self.assertEqual(inv["gst_paise"], gst(inv["taxable_paise"]))

    def test_amounts_are_whole_paise(self):
        inv = build_invoice(quote(), number="EKI/2026-27/000001", issued_at=NOW, buyer=BUYER, seller=SELLER)
        for key in ("taxable_paise", "gst_paise", "total_paise"):
            self.assertIs(type(inv[key]), int)

    def test_indian_digit_grouping(self):
        self.assertEqual(rupees(11_800_000), chr(0x20B9) + "1,18,000.00")


class Numbers(unittest.TestCase):
    def test_the_financial_year_runs_april_to_march(self):
        self.assertEqual(financial_year(datetime(2026, 11, 19)), "2026-27")
        self.assertEqual(financial_year(datetime(2027, 3, 31)), "2026-27")
        self.assertEqual(financial_year(datetime(2027, 4, 1)), "2027-28")
        self.assertEqual(financial_year(datetime(2026, 4, 1)), "2026-27")
        self.assertEqual(financial_year(datetime(2026, 3, 31)), "2025-26")

    def test_invoice_numbers(self):
        self.assertEqual(invoice_number(123, NOW), "EKI/2026-27/000123")
        for bad in (0, -1, 1.5, "1", None):
            with self.subTest(bad=bad), self.assertRaises(InvoiceError):
                invoice_number(bad, NOW)


class Gstin(unittest.TestCase):
    def test_a_correct_gstin_passes_and_changes_fail(self):
        good = valid_gstin()
        self.assertTrue(gstin_is_valid(good))
        self.assertFalse(gstin_is_valid(good[:-1] + ("A" if good[-1] != "A" else "B")))  # wrong check character
        for bad in ("", "27AAAPA1234A1Z", "27aaapa1234a1z5", "XX" + good[2:], None, 5, good + "1"):
            with self.subTest(bad=bad):
                self.assertFalse(gstin_is_valid(bad))

    def test_party_refuses_an_invalid_gstin_or_empty_name(self):
        with self.assertRaises(InvoiceError):
            Party("X", gstin="27AAAPA1234A1Z0")
        with self.assertRaises(InvoiceError):
            Party("  ")
        self.assertEqual(Party("X", gstin=valid_gstin()).gstin, valid_gstin())


class BuildRefusals(unittest.TestCase):
    def test_a_tampered_quote_gets_no_invoice(self):
        q = quote()
        q["gst_paise"] -= 100
        with self.assertRaises(InvoiceError):
            build_invoice(q, number="n", issued_at=NOW, buyer=BUYER, seller=SELLER)

    def test_a_quote_that_does_not_add_up_gets_no_invoice(self):
        from ekido_core.canonical import document_hash

        q = quote()
        q["total_paise"] += 1
        q["terms_hash"] = document_hash({k: v for k, v in q.items() if k != "terms_hash"})  # valid hash, wrong sum
        with self.assertRaises(InvoiceError):
            build_invoice(q, number="n", issued_at=NOW, buyer=BUYER, seller=SELLER)

    def test_an_expired_but_intact_quote_can_still_be_invoiced(self):
        late = datetime(2027, 6, 1, tzinfo=timezone.utc)  # long after the quote expired: the customer paid in time
        inv = build_invoice(quote(), number="n", issued_at=late, buyer=BUYER, seller=SELLER)
        self.assertEqual(inv["issued_at"][:4], "2027")

    def test_garbage_quotes_are_refused(self):
        for bad in ({}, {"terms_hash": "x"}, {"created_at": "nope"}):
            with self.subTest(bad=bad), self.assertRaises(InvoiceError):
                build_invoice(bad, number="n", issued_at=NOW, buyer=BUYER, seller=SELLER)

    def test_the_quote_is_not_changed(self):
        q = quote()
        before = copy.deepcopy(q)
        build_invoice(q, number="n", issued_at=NOW, buyer=BUYER, seller=SELLER)
        self.assertEqual(q, before)


class Pdf(unittest.TestCase):
    def make(self, **kw):
        q = quote(400, distill=True)
        inv = build_invoice(q, number="EKI/2026-27/000042", issued_at=NOW, buyer=kw.pop("buyer", BUYER), seller=SELLER, **kw)
        return q, inv, render_pdf(inv)

    def test_a_valid_small_pdf_with_every_figure(self):
        q, inv, pdf = self.make()
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertTrue(pdf.rstrip().endswith(b"%%EOF"))
        self.assertLess(len(pdf), 60_000)
        text = pdf.decode("latin-1")
        for needle in ("EKI/2026-27/000042", "TEST INVOICE: not a tax invoice", "Sharma Traders", "GST \\(18%\\)",  # the PDF escapes brackets
                       "Rs " + rupees(inv["total_paise"])[1:], "Rs " + rupees(inv["gst_paise"])[1:], q["terms_hash"], "Generated training examples"):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_a_real_invoice_says_tax_invoice_and_a_buyer_gstin_is_printed(self):
        buyer = Party("Sharma Traders", ("Pune",), valid_gstin())
        _, _, pdf = self.make(test_mode=False, buyer=buyer)
        text = pdf.decode("latin-1")
        self.assertIn("TAX INVOICE", text)
        self.assertNotIn("TEST INVOICE", text)
        self.assertIn(valid_gstin(), text)

    def test_text_outside_latin1_does_not_break_the_invoice(self):
        sharma = "".join(chr(c) for c in (0x0936, 0x0930, 0x094D, 0x092E, 0x093E))  # Devanagari "Sharma"
        pune = "".join(chr(c) for c in (0x092A, 0x0941, 0x0923, 0x0947))
        _, _, pdf = self.make(buyer=Party(sharma + " Traders", (pune,)))
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIn("Bill to: ????? Traders", pdf.decode("latin-1"))  # each unsupported letter became a question mark

    def test_the_same_invoice_renders_to_the_same_bytes(self):
        _, inv, first = self.make()
        self.assertEqual(render_pdf(inv), render_pdf(inv))
        self.assertEqual(first, render_pdf(inv))


if __name__ == "__main__":
    unittest.main()

