import unittest
from etl_pipeline.utils import transform

class TestTransformFunctions(unittest.TestCase):

    def test_clean_text(self):
        text = "RT @user123 Check this out! #awesome http://example.com 1234\nNew line."
        expected = "Check this out New line"
        self.assertEqual(transform.clean_text(text), expected)

        text = "@user #hashtag 1234!!! $$$"
        expected = ""
        self.assertEqual(transform.clean_text(text), expected)

        text = "Hello world"
        expected = "Hello world"
        self.assertEqual(transform.clean_text(text), expected)

    def test_lowercase_text(self):
        text = "HeLLo WoRLD"
        expected = "hello world"
        self.assertEqual(transform.lowercase_text(text), expected)

    def test_lemmatize_text(self):
        text = "running runs ran easily the and"
        expected_words = ["running", "run", "ran", "easily"]
        result = transform.lemmatize_text(text).split()
        self.assertNotIn("the", result)
        self.assertNotIn("and", result)
        for word in expected_words:
            self.assertIn(word, result)

    def test_tokenize_text(self):
        text = "Hello, world! This is a test: does it work?"
        expected = ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', ':', 'does', 'it', 'work', '?']
        self.assertEqual(transform.tokenize_text(text), expected)

    def test_detokenize_text(self):
        tokens = ['Hello', ',', 'world', '!']
        expected = "Hello , world !"
        self.assertEqual(transform.detokenize_text(tokens), expected)

    def test_preprocess_text(self):
        text = "RT @user123 Running runs ran easily! Check http://example.com #fun 123"
        processed = transform.preprocess_text(text)
        self.assertNotIn("@user123", processed)
        self.assertNotIn("#fun", processed)
        self.assertNotIn("rt", processed)  
        self.assertNotIn("http", processed)
        self.assertNotIn("123", processed)
        self.assertEqual(processed, processed.lower())
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)

