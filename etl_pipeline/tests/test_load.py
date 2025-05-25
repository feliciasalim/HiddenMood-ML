import os
import unittest
import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from utils.load import load_csv

class TestLoadCSV(unittest.TestCase):
    def test_success(self):
        # Test with valid DataFrame
        data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        captured_output = StringIO()
        with redirect_stdout(captured_output):
            load_csv(data)
        
        # Verify success message
        self.assertIn("Data Loaded to CSV Successfully!", captured_output.getvalue())
        # Verify file creation and content
        self.assertTrue(os.path.exists('mental_health_dataset.csv'))
        df = pd.read_csv('mental_health_dataset.csv')
        pd.testing.assert_frame_equal(df, data)
    
    def test_failure(self):
        # Test with invalid input (string instead of DataFrame)
        invalid_data = "invalid_string_input"
        captured_output = StringIO()
        with redirect_stdout(captured_output):
            load_csv(invalid_data)
        
        # Check that error message is printed
        self.assertIn("Error:", captured_output.getvalue())

