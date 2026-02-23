import chromadb
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import load_dotenv
import re
import numpy as np

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── Wikipedia Topics ─────────────────────────────────────────────────────────

INDIA_WIKIPEDIA_TOPICS = [
    ("Systematic_investment_plan", "india"),
    ("Public_Provident_Fund_(India)", "india"),
    ("Public_Provident_Fund", "india"),
    ("Equity_Linked_Savings_Scheme", "india"),
    ("National_Pension_System", "india"),
    ("Mutual_funds_in_India", "india"),
    ("Income_tax_in_India", "india"),
    ("Nifty_50", "india"),
    ("Reserve_Bank_of_India", "india"),
    ("Zero-based_budgeting", "both"),
    ("401(k)", "us"),
    ("Roth_IRA", "us"),
    ("Index_fund", "us"),
    ("S%26P_500", "us"),
    ("Exchange-traded_fund", "us"),
    ("High-yield_savings_account", "us"),
    ("Dollar_cost_averaging", "both"),
    ("Asset_allocation", "both"),
    ("Compound_interest", "both"),
    ("Inflation", "both"),
    ("Emergency_fund", "both"),
    ("Diversification_(finance)", "both"),
    ("Time_value_of_money", "both"),
    ("Personal_finance", "both"),
    ("Expense_ratio", "both"),
    ("Net_worth", "both"),
]

INVESTOPEDIA_URLS = [
    # ── URLs ────────────────────────────────────────────────────────────────
    {"url": "https://www.investopedia.com/terms/i/indexfund.asp",
     "country": "us", "title": "Index Fund Explained"},
    {"url": "https://www.investopedia.com/terms/r/rothira.asp",
     "country": "us", "title": "Roth IRA Guide"},
    {"url": "https://www.investopedia.com/terms/1/401kplan.asp",
     "country": "us", "title": "401k Plan Guide"},
    {"url": "https://www.investopedia.com/terms/c/compoundinterest.asp",
     "country": "both", "title": "Compound Interest"},
    {"url": "https://www.investopedia.com/terms/d/dollarcostaveraging.asp",
     "country": "both", "title": "Dollar Cost Averaging"},
    {"url": "https://www.investopedia.com/terms/a/assetallocation.asp",
     "country": "both", "title": "Asset Allocation"},
    {"url": "https://www.investopedia.com/terms/i/inflation.asp",
     "country": "both", "title": "Inflation Explained"},
    {"url": "https://www.investopedia.com/terms/p/personalfinance.asp",
     "country": "both", "title": "Personal Finance Basics"},
    {"url": "https://www.investopedia.com/terms/b/budget.asp",
     "country": "both", "title": "Budgeting Guide"},
    {"url": "https://www.investopedia.com/terms/n/networth.asp",
     "country": "both", "title": "Net Worth Explained"},
    {"url": "https://www.investopedia.com/terms/m/mutualfund.asp",
     "country": "both", "title": "Mutual Fund Guide"},
    {"url": "https://www.investopedia.com/terms/d/diversification.asp",
     "country": "both", "title": "Diversification Strategy"},
    {"url": "https://www.investopedia.com/terms/r/rebalancing.asp",
     "country": "both", "title": "Portfolio Rebalancing"},
    {"url": "https://www.investopedia.com/terms/e/etf.asp",
     "country": "us", "title": "ETF Exchange Traded Fund"},
    {"url": "https://www.investopedia.com/terms/f/four-percent-rule.asp",
     "country": "us", "title": "Four Percent Retirement Rule"},
    {"url": "https://www.investopedia.com/terms/s/savings-rate.asp",
     "country": "both", "title": "Savings Rate Guide"},
    {"url": "https://www.investopedia.com/terms/s/systematic-investment-plan.asp",
     "country": "india", "title": "Systematic Investment Plan"},
    {"url": "https://www.investopedia.com/terms/e/emergency_fund.asp",
     "country": "both", "title": "Emergency Fund"},
    {"url": "https://www.investopedia.com/terms/p/public-provident-fund.asp",
     "country": "india", "title": "Public Provident Fund India"},
    {"url": "https://www.investopedia.com/high-yield-savings-account-4770140",
     "country": "us", "title": "High Yield Savings Account Guide"},
    {"url": "https://www.investopedia.com/terms/t/taxdeferred.asp",
     "country": "both", "title": "Tax Deferred Investments"},
    {"url": "https://www.investopedia.com/terms/c/cola.asp",
     "country": "both", "title": "Cost of Living Adjustment"},
    {"url": "https://www.investopedia.com/vanguard-review-4587389",
     "country": "us", "title": "Vanguard Funds Review"},
    {"url": "https://www.investopedia.com/terms/e/expenseratio.asp",
     "country": "both", "title": "Expense Ratio Guide"},
    {"url": "https://www.investopedia.com/budgeting-saving-4427755",
     "country": "both", "title": "Budgeting and Saving Guide"},
    {"url": "https://www.investopedia.com/personal-finance/",
     "country": "both", "title": "Personal Finance Hub"},
]


# ── Fetchers ─────────────────────────────────────────────────────────────────

def fetch_wikipedia(topic: str) -> str:
    """
    Fetches full Wikipedia article by scraping HTML directly.
    Gets complete article content instead of just the summary.
    """
    try:
        clean_topic = topic.replace("%26", "%26").replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{clean_topic}"
        headers = {"User-Agent": "FinSense-Educational-App/1.0 (Educational project)"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            return ""

        paragraphs = content_div.find_all("p")
        text_parts = []
        for p in paragraphs[:20]:
            text = p.get_text()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 50:
                text_parts.append(text)

        full_text = " ".join(text_parts)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return full_text[:6000] if len(full_text) > 300 else ""

    except Exception as e:
        print(f"  Wikipedia failed for {topic}: {e}")
        return ""


def fetch_investopedia(url: str) -> str:
    """Fetches article text from Investopedia with realistic browser headers."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form", "button"]):
            tag.decompose()

        article = (soup.find("article") or
                   soup.find("div", {"id": "article-body"}) or
                   soup.find("div", {"class": re.compile("article-body")}) or
                   soup.find("div", {"class": re.compile("comp-article")}) or
                   soup.find("main"))

        if article:
            text = article.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:6000] if len(text) > 500 else ""

    except Exception as e:
        print(f"  Investopedia failed for {url}: {e}")
        return ""


# ── Semantic Chunking ─────────────────────────────────────────────────────────

def semantic_chunk(text: str, embedder: SentenceTransformer) -> list:
    """Semantic chunking — splits text at points where meaning changes."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 3:
        return [text] if len(text) > 100 else []

    embeddings = embedder.encode(sentences)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    threshold = np.percentile(similarities, 25)
    split_points = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    chunks = []
    prev = 0
    for split in split_points:
        chunk_text = " ".join(sentences[prev:split])
        if len(chunk_text) > 100:
            chunks.append(chunk_text)
        prev = split

    final_chunk = " ".join(sentences[prev:])
    if len(final_chunk) > 100:
        chunks.append(final_chunk)

    merged_chunks = []
    buffer = ""
    for chunk in chunks:
        buffer = (buffer + " " + chunk).strip()
        if len(buffer.split()) >= 100:
            merged_chunks.append(buffer)
            buffer = ""

    if buffer and len(buffer) > 100:
        if merged_chunks:
            merged_chunks[-1] += " " + buffer
        else:
            merged_chunks.append(buffer)

    return merged_chunks


def recursive_chunk(text: str) -> list:
    """Recursive character text splitter — fallback chunking method."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c) > 100]


def smart_chunk(text: str, embedder: SentenceTransformer) -> list:
    """Semantic first, recursive fallback."""
    semantic_chunks = semantic_chunk(text, embedder)
    if len(semantic_chunks) < 3 and len(text) > 1000:
        return recursive_chunk(text)
    return semantic_chunks if semantic_chunks else recursive_chunk(text)


# ── Build Knowledge Base ──────────────────────────────────────────────────────

def build_knowledge_base():
    """Builds ChromaDB knowledge base from Wikipedia + Investopedia."""
    print("=" * 60)
    print("Building FinSense RAG Knowledge Base")
    print("Chunking: Semantic + Recursive fallback")
    print("Sources: Wikipedia + Investopedia")
    print("=" * 60)

    chroma_path = Path(__file__).resolve().parent.parent / "chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        chroma_client.delete_collection("financial_literacy")
        print("Cleared existing knowledge base\n")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="financial_literacy",
        metadata={"description": "FinSense real-time financial knowledge base"}
    )

    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded\n")

    all_chunks = []
    all_metadatas = []
    all_ids = []
    doc_id = 0
    success_count = 0

    # ── Wikipedia ────────────────────────────────────────────────────────────
    print("── Wikipedia Articles ──")
    for topic, country in INDIA_WIKIPEDIA_TOPICS:
        display = topic.replace("_", " ").replace("%26", "&")
        print(f"  Fetching: {display}...")
        content = fetch_wikipedia(topic)

        if content and len(content) > 300:
            chunks = smart_chunk(content, embedder)
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "title": display,
                    "country": country,
                    "source": "Wikipedia",
                    "type": "wikipedia"
                })
                all_ids.append(f"wiki_{doc_id}_{j}")
            doc_id += 1
            success_count += 1
            print(f"  ✅ {len(chunks)} semantic chunks — {display}")
        else:
            print(f"  ⚠️ Skipped — {display}")

    # ── Investopedia ─────────────────────────────────────────────────────────
    print("\n── Investopedia Articles ──")
    for item in INVESTOPEDIA_URLS:
        print(f"  Fetching: {item['title']}...")
        content = fetch_investopedia(item["url"])

        if content and len(content) > 300:
            chunks = smart_chunk(content, embedder)
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "title": item["title"],
                    "country": item["country"],
                    "source": "Investopedia",
                    "type": "investopedia"
                })
                all_ids.append(f"investo_{doc_id}_{j}")
            doc_id += 1
            success_count += 1
            print(f"  ✅ {len(chunks)} semantic chunks — {item['title']}")
        else:
            print(f"  ⚠️ Skipped — {item['title']}")

    if not all_chunks:
        print("\n❌ No content fetched — check internet connection")
        return

    # ── Generate Embeddings and Store ────────────────────────────────────────
    print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
    batch_size = 64
    all_embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print(f"\n{'=' * 60}")
    print(f"✅ Knowledge Base Built Successfully!")
    print(f"   Documents processed: {success_count}")
    print(f"   Total chunks stored: {len(all_chunks)}")
    print(f"   India chunks: {len([m for m in all_metadatas if m['country'] in ['india', 'both']])}")
    print(f"   US chunks: {len([m for m in all_metadatas if m['country'] in ['us', 'both']])}")
    print(f"   Chunking method: Semantic + Recursive fallback")
    print(f"   Stored in: backend/chroma_db/")
    print(f"{'=' * 60}")
    print("\nRe-run anytime to refresh with latest content!")


if __name__ == "__main__":
    build_knowledge_base()