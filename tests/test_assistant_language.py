import unittest

from ekido_platform.assistant_language import detect_language


class Detect(unittest.TestCase):
    def test_devanagari_is_hindi_even_with_english_words(self):
        self.assertEqual(detect_language("मेरे invoice से total निकालना है"), "hi")

    def test_hindi_in_english_letters_is_hinglish(self):
        for text in ("grahak ke message alag karne hain", "dukaan ka hisaab rakhna hai", "is bot ko Hindi me jawab dene wala banao"):
            with self.subTest(text=text):
                self.assertEqual(detect_language(text), "hinglish")

    def test_english_stays_english_even_with_a_lookalike_word(self):
        for text in ("Do you have a bot for me?", "I sell banana chips online", "read my invoices", "chatbot", ""):
            with self.subTest(text=text):
                self.assertEqual(detect_language(text), "en")


if __name__ == "__main__":
    unittest.main()
