from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── Keywords ─────────────────────────────────────────────────────────────────

INDIA_KEYWORDS = '"mutual fund" OR "stock market" OR "RBI" OR "SEBI" OR "Nifty" OR "SIP" OR "personal finance India"'

US_KEYWORDS = '"Federal Reserve" OR "S&P 500" OR "stock market" OR "inflation" OR "401k" OR "personal finance" OR "investing"'


def get_financial_news(country: str, max_articles: int = 5) -> dict:
    """
    Fetches live financial news headlines filtered by country.
    """
    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        keywords = INDIA_KEYWORDS if country == "india" else US_KEYWORDS

        response = newsapi.get_everything(
            q=keywords,
            language="en",
            sort_by="relevancy",
            from_param=from_date,
            to=to_date,
            page_size=20
        )

        # Clean and filter articles
        articles = []
        for article in response.get("articles", []):
            title = article.get("title", "")
            description = article.get("description", "")

            # Skip if no content
            if not title or not description:
                continue

            # Skip removed articles
            if title == "[Removed]":
                continue

            # Skip clearly non-financial articles
            sports_keywords = ["olympics", "basketball", "football", "soccer",
                              "hockey", "baseball", "nba", "nfl", "mlb", "nhl"]
            if any(kw in title.lower() for kw in sports_keywords):
                continue

            articles.append({
                "title": title,
                "description": description,
                "source": article["source"]["name"],
                "url": article["url"],
                "published_at": article["publishedAt"],
            })

            if len(articles) >= max_articles:
                break

        return {
            "country": country,
            "articles": articles,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        return {
            "country": country,
            "articles": [],
            "error": str(e),
            "status": "fallback"
        }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing India news...")
    india_news = get_financial_news("india")
    for article in india_news["articles"]:
        print(f"- {article['title']} ({article['source']})")

    print("\nTesting US news...")
    us_news = get_financial_news("us")
    for article in us_news["articles"]:
        print(f"- {article['title']} ({article['source']})")