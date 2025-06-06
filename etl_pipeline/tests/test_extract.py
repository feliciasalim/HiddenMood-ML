import unittest
from unittest.mock import patch, MagicMock
from utils.extract import scrape_reddit

class TestRedditScraper(unittest.TestCase):

    @patch('etl_pipeline.utils.extract.praw.Reddit')
    def test_scrape_reddit(self, mock_reddit_class):
        mock_reddit = MagicMock()
        mock_subreddit = MagicMock()
        mock_submission = MagicMock()

        mock_submission.id = 'abc123'
        mock_submission.title = 'Test Title'
        mock_submission.selftext = 'This is a valid ASCII test post with proper length.'

        mock_subreddit.search.return_value = [mock_submission]
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_class.return_value = mock_reddit

        results = scrape_reddit(
            client_id='dummy_id',
            client_secret='dummy_secret',
            username='dummy_user',
            password='dummy_pass',
            user_agent='dummy_agent',
            keywords=['test']
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 'abc123')
        self.assertEqual(results[0]['title'], 'Test Title')
        self.assertIn('valid ASCII test', results[0]['selftext'])

