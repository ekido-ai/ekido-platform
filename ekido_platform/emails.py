"""The words and layout of the emails the Platform sends. Senders (v1_accounts.py) only deliver them.

An invite names the team and who sent it, so the person has a reason to trust it, and has one clear button. It goes
out as HTML with a plain-text part: the HTML uses tables and inline styles only, since many mail apps drop <style>,
and every value from a person (team name, inviter) is escaped, kept to one line and length-capped.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

LINK_MINUTES = 15

ROLE_WORDS = {
    "owner": ("an owner", "manage the team, its billing and API keys, and train models"),
    "admin": ("an admin", "invite people, manage API keys and train models"),
    "member": ("a member", "see the team's jobs and train models"),
}

INK, INK_2, MUTED, LINE, PAPER, KESARI = "#131E29", "#44525D", "#6B7882", "#D9D9D6", "#F7F5F2", "#FE5000"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class Invite:
    team: str
    inviter: str
    role: str
    site_url: str


def one_line(value: str, limit: int) -> str:
    """Control characters and runs of whitespace become one space; long values end in an ellipsis."""
    value = re.sub(r"[\x00-\x1f\x7f\s]+", " ", value or "").strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def invite_subject(invite: Invite) -> str:
    return f"{one_line(invite.inviter, 60)} invited you to {one_line(invite.team, 60)} on Ekido"


def invite_text(invite: Invite, link: str) -> str:
    team, inviter = one_line(invite.team, 80), one_line(invite.inviter, 120)
    article, can = ROLE_WORDS.get(invite.role, ROLE_WORDS["member"])
    login = f"{invite.site_url.rstrip('/')}/login"
    return (
        f"{inviter} invited you to join {team} on Ekido.\n\n"
        f"You'll join as {article}: you can {can}.\n\n"
        f"Join {team}:\n{link}\n\n"
        f"The button signs you in and adds you to the team. It works once, for {LINK_MINUTES} minutes. "
        f"If it has expired, sign in at {login} with this email address and you'll join then.\n\n"
        "Ekido trains small AI models of your own, on devices people already have.\n\n"
        f"Weren't expecting this? You can ignore it: nothing happens unless you open the link."
    )


def invite_html(invite: Invite, link: str) -> str:
    e = html.escape
    team, inviter = e(one_line(invite.team, 80)), e(one_line(invite.inviter, 120))
    article, can = ROLE_WORDS.get(invite.role, ROLE_WORDS["member"])
    href, site = e(link, quote=True), invite.site_url.rstrip("/")
    login = e(f"{site}/login", quote=True)
    initial = e(one_line(invite.team, 80)[:1].upper() or "E")
    preheader = e(f"Join {one_line(invite.team, 80)} on Ekido. The link works for {LINK_MINUTES} minutes.")
    steps = "".join(
        f'<tr><td style="padding:0 0 10px 0;vertical-align:top;width:28px;">'
        f'<div style="width:22px;height:22px;border-radius:11px;background:{PAPER};border:1px solid {LINE};color:{INK};'
        f'font:600 12px/22px {FONT};text-align:center;">{n}</div></td>'
        f'<td style="padding:0 0 10px 8px;font:14px/22px {FONT};color:{INK_2};">{text}</td></tr>'
        for n, text in (
            (1, f"Press <strong style=\"color:{INK};\">Join {team}</strong>."),
            (2, "You're signed in with this email address. No password to make."),
            (3, f"You land in Ekido as {e(article)} of {team}."),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>{e(invite_subject(invite))}</title></head>
<body style="margin:0;padding:0;background:{PAPER};">
<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">{preheader}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{PAPER};padding:32px 12px;">
<tr><td align="center">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:520px;background:#FFFFFF;border:1px solid {LINE};border-radius:16px;overflow:hidden;">
<tr><td style="height:4px;background:{KESARI};font-size:0;line-height:0;">&nbsp;</td></tr>
<tr><td style="padding:28px 32px 0 32px;font:700 18px/24px {FONT};color:{INK};letter-spacing:-0.01em;">Ekido</td></tr>
<tr><td style="padding:24px 32px 0 32px;">
  <table role="presentation" cellpadding="0" cellspacing="0"><tr>
    <td style="width:48px;height:48px;border-radius:12px;background:{INK};color:#FFFFFF;font:700 22px/48px {FONT};text-align:center;">{initial}</td>
    <td style="padding-left:14px;font:13px/18px {FONT};color:{MUTED};">You're invited to<br><span style="font:600 16px/22px {FONT};color:{INK};">{team}</span></td>
  </tr></table>
</td></tr>
<tr><td style="padding:22px 32px 0 32px;font:700 24px/30px {FONT};color:{INK};letter-spacing:-0.02em;">{inviter} invited you to join {team}</td></tr>
<tr><td style="padding:10px 32px 0 32px;font:15px/24px {FONT};color:{INK_2};">You'll join as <strong style="color:{INK};">{e(article)}</strong>: you can {e(can)}.</td></tr>
<tr><td style="padding:26px 32px 0 32px;">
  <table role="presentation" cellpadding="0" cellspacing="0"><tr>
    <td style="border-radius:999px;background:{INK};">
      <a href="{href}" style="display:inline-block;padding:13px 26px;font:600 15px/20px {FONT};color:#FFFFFF;text-decoration:none;border-radius:999px;">Join {team} &rarr;</a>
    </td>
  </tr></table>
</td></tr>
<tr><td style="padding:8px 32px 0 32px;font:12px/18px {FONT};color:{MUTED};">Works once, for {LINK_MINUTES} minutes.</td></tr>
<tr><td style="padding:26px 32px 0 32px;"><div style="border-top:1px solid {LINE};font-size:0;line-height:0;">&nbsp;</div></td></tr>
<tr><td style="padding:18px 32px 6px 32px;font:600 13px/18px {FONT};color:{INK};">What happens next</td></tr>
<tr><td style="padding:6px 32px 0 32px;"><table role="presentation" cellpadding="0" cellspacing="0" width="100%">{steps}</table></td></tr>
<tr><td style="padding:10px 32px 0 32px;font:13px/20px {FONT};color:{INK_2};">Link expired? <a href="{login}" style="color:{INK};text-decoration:underline;">Sign in</a> with this email address and you'll join then.</td></tr>
<tr><td style="padding:16px 32px 0 32px;font:12px/18px {FONT};color:{MUTED};">If the button doesn't work, paste this into your browser:<br><a href="{href}" style="color:{INK_2};word-break:break-all;">{e(link)}</a></td></tr>
<tr><td style="padding:24px 32px 28px 32px;font:12px/18px {FONT};color:{MUTED};">Weren't expecting this? You can ignore it: nothing happens unless you open the link.</td></tr>
</table>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:520px;">
<tr><td style="padding:16px 8px 0 8px;font:12px/18px {FONT};color:{MUTED};text-align:center;">Ekido trains small AI models of your own, on devices people already have.<br>
<a href="{e(site, quote=True)}" style="color:{MUTED};">{e(re.sub(r'^https?://', '', site))}</a></td></tr>
</table>
</td></tr></table>
</body></html>"""
