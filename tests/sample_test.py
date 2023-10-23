import unittest


class TestExample(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(2 + 2, 4, 'Sample test, always true')
