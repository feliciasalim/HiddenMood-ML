import praw
import time

def scrape_reddit(client_id, client_secret, username, password, user_agent, keywords):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        username=username,
        password=password,
        user_agent=user_agent
    )

    results = []
    seen_ids = set()
    completed_keywords = set()

    for keyword in keywords:
        print(f"Searching: {keyword}")
        try:
            for submission in reddit.subreddit('all').search(keyword, sort='new'):
                if submission.id in seen_ids:
                    continue
                selftext = submission.selftext.strip()
                if not selftext or not (30 <= len(selftext) <= 300) or not selftext.isascii():
                    continue
                results.append({
                    'id': submission.id,
                    'title': submission.title,
                    'selftext': selftext
                })
                seen_ids.add(submission.id)
                print(f"Total collected: {len(results)}")
                time.sleep(1)
        except Exception as e:
            print(f"Error for '{keyword}': {e}")
        completed_keywords.add(keyword)
        print(f"Done with keyword: {keyword}")

    print("Scraping done.")
    return results
