import unittest

from ekido_platform.v1_assistant import RefusalBreaker


class Breaker(unittest.TestCase):
    def test_it_opens_after_the_limit_and_closes_after_the_window(self):
        breaker = RefusalBreaker(limit=3, window=60)
        for second in range(2):
            breaker.record("a", now=second)
        self.assertFalse(breaker.open("a", now=2))
        breaker.record("a", now=2)
        self.assertTrue(breaker.open("a", now=3))
        self.assertFalse(breaker.open("b", now=3))  # one person's refusals do not pause another
        self.assertFalse(breaker.open("a", now=100))


if __name__ == "__main__":
    unittest.main()
