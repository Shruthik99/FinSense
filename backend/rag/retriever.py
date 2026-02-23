import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── Initialize ChromaDB and Embedder ─────────────────────────────────────────

chroma_path = Path(__file__).resolve().parent.parent / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_knowledge(
    query: str,
    country: str,
    n_results: int = 4
) -> list:
    """
    Retrieves the most relevant financial knowledge chunks for a query.
    Filters by country so India users get India-relevant advice
    and US users get US-relevant advice.

    Args:
        query: The question or topic to search for
        country: "india" or "us"
        n_results: Number of chunks to return

    Returns:
        List of dicts with content, title, source, country
    """
    try:
        collection = chroma_client.get_collection("financial_literacy")

        # Embed the query
        query_embedding = embedder.encode(query).tolist()

        # Filter by country — include "both" tagged content always
        where_filter = {
            "$or": [
                {"country": {"$eq": country}},
                {"country": {"$eq": "both"}}
            ]
        }

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        knowledge = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = round(1 - dist, 4)

            knowledge.append({
                "content": doc,
                "title": meta.get("title", "Unknown"),
                "source": meta.get("source", "Unknown"),
                "country": meta.get("country", "both"),
                "similarity": similarity
            })

        # Sort by similarity — most relevant first
        knowledge.sort(key=lambda x: x["similarity"], reverse=True)
        return knowledge

    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []


def retrieve_for_anomalies(
    anomalies: list,
    country: str,
    n_results: int = 3
) -> list:
    """
    Retrieves knowledge specifically targeting the user's worst anomalies.
    Called during agent reasoning to ground advice in real sources.
    """
    if not anomalies:
        return retrieve_knowledge("personal finance budgeting basics", country, n_results)

    # Build query from worst anomalies
    critical = [a for a in anomalies if a["verdict"] == "critical"]
    warning = [a for a in anomalies if a["verdict"] == "warning"]

    worst = (critical + warning)[:3]

    if worst:
        categories = [a["category"].replace("_", " ") for a in worst]
        query = f"budgeting advice overspending {' '.join(categories)} savings investment"
    else:
        query = "budgeting savings investment personal finance"

    return retrieve_knowledge(query, country, n_results)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Testing RAG Retriever")
    print("=" * 50)

    # Test 1 — India query
    print("\nTest 1 — India: 'what is SIP and how to start'")
    results = retrieve_knowledge("what is SIP and how to start investing", "india", n_results=3)
    for r in results:
        print(f"\n  📄 {r['title']} ({r['source']}) — similarity: {r['similarity']}")
        print(f"     {r['content'][:150]}...")

    # Test 2 — US query
    print("\n" + "=" * 50)
    print("\nTest 2 — US: 'how does a 401k work'")
    results = retrieve_knowledge("how does a 401k work and should I contribute", "us", n_results=3)
    for r in results:
        print(f"\n  📄 {r['title']} ({r['source']}) — similarity: {r['similarity']}")
        print(f"     {r['content'][:150]}...")

    # Test 3 — Anomaly-based retrieval
    print("\n" + "=" * 50)
    print("\nTest 3 — Anomaly retrieval: overspending on dining and subscriptions")
    mock_anomalies = [
        {"category": "dining_out", "verdict": "critical"},
        {"category": "subscriptions", "verdict": "critical"},
        {"category": "savings", "verdict": "critical"},
    ]
    results = retrieve_for_anomalies(mock_anomalies, "india", n_results=3)
    for r in results:
        print(f"\n  📄 {r['title']} ({r['source']}) — similarity: {r['similarity']}")
        print(f"     {r['content'][:150]}...")