"""Sign-in email senders: SMTP delivery, and a clear 503 (never a 500, never the link or password in the log) when a send fails."""
import smtplib
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from ekido_platform.main import Settings, create_app
from ekido_platform.emails import Invite, invite_html, invite_subject, invite_text
from ekido_platform.v1_accounts import SesEmail, SmtpEmail

SITE = "https://ekido.example"


class FakeSmtp:
    """Records what an SMTP session did; `refuse` makes login fail like a wrong app password."""

    sent: list = []
    steps: list = []
    refuse = False

    def __init__(self, host, port, timeout):
        FakeSmtp.steps.append(("connect", host, port))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        FakeSmtp.steps.append(("quit",))

    def starttls(self):
        FakeSmtp.steps.append(("starttls",))

    def login(self, user, password):
        FakeSmtp.steps.append(("login", user))
        if FakeSmtp.refuse:
            raise smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")

    def send_message(self, message):
        FakeSmtp.sent.append(message)


class RecordingSes:
    def __init__(self):
        self.calls = []

    def send_email(self, **kwargs):
        self.calls.append(kwargs)


class RejectingSes:
    def send_email(self, **kwargs):
        raise RuntimeError("MessageRejected: Email address is not verified")


def client_with(sender):
    # create_app builds the "log" sender by calling LogEmail(); hand it the sender under test instead.
    with mock.patch("ekido_platform.main.LogEmail", lambda: sender):
        app = create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE,
                                  cors_origins=[SITE], session_cookie_secure=False))
    return TestClient(app)


class Senders(unittest.TestCase):
    def setUp(self):
        FakeSmtp.sent, FakeSmtp.steps, FakeSmtp.refuse = [], [], False

    def smtp(self):
        return SmtpEmail("smtp.example", 587, "team@example.com", "app-password", "Ekido <team@example.com>", smtp_class=FakeSmtp)

    def test_smtp_sends_over_starttls_after_login(self):
        self.smtp().send_login_link("asha@example.com", f"{SITE}/login/verify?token=abc")
        self.assertEqual(FakeSmtp.steps, [("connect", "smtp.example", 587), ("starttls",), ("login", "team@example.com"), ("quit",)])
        message = FakeSmtp.sent[0]
        self.assertEqual((message["To"], message["From"], message["Subject"]), ("asha@example.com", "Ekido <team@example.com>", "Your Ekido sign-in link"))
        self.assertIn(f"{SITE}/login/verify?token=abc", message.get_content())

    def test_a_refused_smtp_send_is_a_clear_503_and_logs_no_secrets(self):
        FakeSmtp.refuse = True
        client = client_with(self.smtp())
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            response = client.post("/v1/auth/email/start", json={"email": "asha@example.com"})
        self.assertEqual((response.status_code, response.json()["error"]), (503, "EMAIL_NOT_SENT"))
        self.assertIn("couldn't send the email", response.json()["detail"])
        joined = " ".join(logs.output)
        self.assertIn("SMTPAuthenticationError", joined)
        self.assertNotIn("token=", joined)
        self.assertNotIn("app-password", joined)

    def test_an_ses_rejection_is_a_clear_503(self):
        client = client_with(SesEmail("Ekido <team@example.com>", client=RejectingSes()))
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING"):
            response = client.post("/v1/auth/email/start", json={"email": "someone@example.com"})
        self.assertEqual((response.status_code, response.json()["error"]), (503, "EMAIL_NOT_SENT"))

    def test_an_invite_names_the_team_and_inviter_and_has_one_button(self):
        invite = Invite(team="Asha's Clinic", inviter="asha@example.com", role="admin", site_url=SITE)
        link = f"{SITE}/login/verify?token=abc"
        self.assertEqual(invite_subject(invite), "asha@example.com invited you to Asha's Clinic on Ekido")
        text = invite_text(invite, link)
        for part in ("asha@example.com invited you to join Asha's Clinic", "as an admin", link, f"{SITE}/login", "15 minutes"):
            self.assertIn(part, text)
        page = invite_html(invite, link)
        self.assertEqual(page.count(f'href="{link}"'), 2)  # the button, and the paste-this-link fallback
        self.assertIn("Join Asha&#x27;s Clinic", page)
        self.assertIn("an admin", page)

    def test_an_invite_escapes_what_people_typed(self):
        invite = Invite(team='<script>alert(1)</script>\nBcc: x@evil.co', inviter='Eve "<b>"', role="member", site_url=SITE)
        page = invite_html(invite, f"{SITE}/login/verify?token=abc")
        self.assertNotIn("<script>", page)
        self.assertNotIn("<b>", page)
        self.assertNotIn("\n", invite_subject(invite))
        self.assertNotIn("\nBcc", invite_text(invite, "x"))

    def test_smtp_sends_an_invite_as_text_and_html(self):
        invite = Invite(team="Shop", inviter="owner@x.co", role="member", site_url=SITE)
        self.smtp().send_invite("staff@x.co", f"{SITE}/login/verify?token=abc", invite)
        message = FakeSmtp.sent[0]
        self.assertEqual(message["Subject"], "owner@x.co invited you to Shop on Ekido")
        self.assertEqual([part.get_content_type() for part in message.iter_parts()], ["text/plain", "text/html"])

    def test_ses_sends_an_invite_as_text_and_html(self):
        ses = RecordingSes()
        SesEmail("Ekido <team@example.com>", client=ses).send_invite("staff@x.co", f"{SITE}/login/verify?token=abc", Invite("Shop", "owner@x.co", "member", SITE))
        message = ses.calls[0]["Message"]
        self.assertEqual(message["Subject"]["Data"], "owner@x.co invited you to Shop on Ekido")
        self.assertEqual(sorted(message["Body"]), ["Html", "Text"])

    def test_smtp_is_chosen_from_settings(self):
        with mock.patch("ekido_platform.main.SmtpEmail") as smtp:
            create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="smtp", smtp_host="smtp.example",
                                smtp_user="team@example.com", smtp_password="x", email_from="Ekido <team@example.com>",
                                site_url=SITE, cors_origins=[SITE], session_cookie_secure=False))
        smtp.assert_called_once_with("smtp.example", 587, "team@example.com", "x", "Ekido <team@example.com>")


if __name__ == "__main__":
    unittest.main()
