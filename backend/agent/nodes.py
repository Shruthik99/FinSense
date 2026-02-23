import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Add backend to path so we can import our modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.state import AgentState
from mcp_tools.market_data import get_market_data
from mcp_tools.inflation import get_inflation
from mcp_tools.news import get_financial_news
from mcp_tools.calculator import generate_projections, calculate_ppf_returns
from mcp_tools.tax_estimator import get_tax_estimate
from ml.anomaly_detector import detect_anomalies
from ml.health_score import compute_health_score

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


def add_step(state: AgentState, step_name: str, detail: str = "") -> list:
    """Helper to add a reasoning step to the state."""
    steps = state.get("steps", [])
    steps.append({
        "step_number": len(steps) + 1,
        "step_name": step_name,
        "detail": detail,
        "timestamp": datetime.now().isoformat(),
        "status": "complete"
    })
    return steps


# ── Node 1: Analyze Spending ─────────────────────────────────────────────────
def node_analyze_spending(state: AgentState) -> AgentState:
    """
    Runs ML anomaly detection on the user's spending.
    Detects which categories are unusual compared to healthy benchmarks.
    """
    print("Node 1: Analyzing spending patterns...")

    spending = state["spending"]
    monthly_income = state["monthly_income"]
    country = state["country"]

    # Run anomaly detection
    anomalies = detect_anomalies(spending, monthly_income, country)

    # Compute health score
    health_result = compute_health_score(spending, monthly_income, anomalies, country)

    steps = add_step(
        state,
        "Analyzing spending patterns",
        f"Found {len([a for a in anomalies if a['is_anomalous']])} anomalies in your budget"
    )

    return {
        **state,
        "anomalies": anomalies,
        "health_score": health_result["score"],
        "health_grade": health_result["grade"],
        "steps": steps
    }


# ── Node 2: Fetch Live Data ──────────────────────────────────────────────────
def node_fetch_live_data(state: AgentState) -> AgentState:
    """
    Calls all MCP tools to fetch real-world data.
    Inflation rate, market prices, news, tax estimate.
    """
    print("Node 2: Fetching live data...")

    country = state["country"]
    monthly_income = state["monthly_income"]
    annual_income = monthly_income * 12

    # Fetch all live data
    inflation_data = get_inflation(country)
    market_data = get_market_data(country)
    news_articles = get_financial_news(country, max_articles=5)
    tax_estimate = get_tax_estimate(country, annual_income)

    # Calculate investment projections
    spending = state["spending"]
    total_spending = sum(v for k, v in spending.items()
                        if k not in ["savings", "investments"])
    investable_amount = max(0, monthly_income - total_spending)
    projections = generate_projections(investable_amount, country)

    steps = add_step(
        state,
        "Fetching live market and economic data",
        f"Inflation: {inflation_data.get('inflation_rate', 'N/A')}% | "
        f"Fetched market prices + {len(news_articles.get('articles', []))} news articles"
    )

    return {
        **state,
        "inflation_data": inflation_data,
        "market_data": market_data,
        "news_articles": news_articles.get("articles", []),
        "tax_estimate": tax_estimate,
        "projections": projections,
        "steps": steps
    }


# ── Node 3: Retrieve Knowledge ───────────────────────────────────────────────
def node_retrieve_knowledge(state: AgentState) -> AgentState:
    """
    Searches the RAG knowledge base for relevant financial guidance.
    Retrieves passages based on the user's anomalies and country.
    """
    print("Node 3: Retrieving financial knowledge...")

    try:
        from rag.retriever import retrieve_knowledge
        anomalies = state["anomalies"]
        country = state["country"]

        # Build query from the worst anomalies
        anomalous = [a for a in anomalies if a["is_anomalous"]]
        if anomalous:
            worst = sorted(anomalous, key=lambda x: x["anomaly_score"], reverse=True)[:3]
            query = f"budgeting advice for overspending on {', '.join([a['category'] for a in worst])}"
        else:
            query = f"general budgeting and investing advice for {country}"

        knowledge = retrieve_knowledge(query, country, n_results=4)

        steps = add_step(
            state,
            "Retrieving financial literacy knowledge",
            f"Found {len(knowledge)} relevant passages from trusted sources"
        )

        return {**state, "retrieved_knowledge": knowledge, "steps": steps}

    except Exception as e:
        # RAG not built yet — continue without it
        steps = add_step(state, "Knowledge retrieval", "Skipped — knowledge base not built yet")
        return {**state, "retrieved_knowledge": [], "steps": steps}


# ── Node 4: Generate Roast ───────────────────────────────────────────────────
def node_generate_roast(state: AgentState) -> AgentState:
    """
    Uses Gemini to generate a brutally honest, funny roast
    based on the user's actual spending anomalies and data.
    """
    print("Node 4: Generating roast...")

    country = state["country"]
    monthly_income = state["monthly_income"]
    spending = state["spending"]
    anomalies = state["anomalies"]
    health_score = state["health_score"]
    inflation_rate = state["inflation_data"].get("inflation_rate", "N/A")
    language = state["language"]
    currency = "₹" if country == "india" else "$"

    # Find the worst spending categories
    anomalous = [a for a in anomalies if a["is_anomalous"]]
    anomaly_text = "\n".join([
        f"- {a['category']}: {currency}{a['amount']:,}/month "
        f"({a['percentage_of_income']}% of income, benchmark is {a['benchmark_percentage']}%)"
        for a in anomalous
    ]) if anomalous else "No major anomalies found"

    hinglish_instruction = "Write in Hinglish (mix of Hindi and English, casual tone)" if language == "hinglish" else "Write in English"

    prompt = f"""You are a brutally honest financial advisor who roasts people's budgets.
Be specific, funny, and harsh but not mean-spirited. Reference their actual numbers.

User's Financial Profile:
- Country: {country.upper()}
- Monthly Income: {currency}{monthly_income:,}
- Financial Health Score: {health_score}/100
- Current Inflation Rate: {inflation_rate}%

Problematic Spending:
{anomaly_text}

{hinglish_instruction}

Write a roast (150-200 words) that:
1. Opens with a punchy one-liner about their overall financial situation
2. Calls out their 2-3 worst spending habits with specific numbers
3. Makes a comparison (e.g. "your Swiggy spend could fund X months of SIP")
4. Ends with a one-liner that stings but motivates

Be specific to their numbers. Do not be generic."""

    response = model.generate_content(prompt)
    roast = response.text

    steps = add_step(state, "Generating your financial roast", "Roast generated successfully")

    return {**state, "roast": roast, "steps": steps}


# ── Node 5: Generate Coach Plan ──────────────────────────────────────────────
def node_generate_coach_plan(state: AgentState) -> AgentState:
    """
    Generates a serious, actionable financial coach plan.
    Uses all fetched data to create personalized advice.
    """
    print("Node 5: Generating coach plan...")

    country = state["country"]
    monthly_income = state["monthly_income"]
    spending = state["spending"]
    anomalies = state["anomalies"]
    inflation_rate = state["inflation_data"].get("inflation_rate", "N/A")
    tax_tip = state["tax_estimate"].get("tip", "")
    projections = state["projections"]
    knowledge = state["retrieved_knowledge"]
    currency = "₹" if country == "india" else "$"
    framework = "40/30/30" if country == "india" else "50/30/20"

    # Build projection summary
    proj_data = projections.get("projections", {})
    proj_summary = "\n".join([
        f"- {k.replace('_', ' ').title()}: {currency}{v.get('future_value', 0):,.0f}"
        for k, v in proj_data.items()
    ]) if proj_data else "Projections unavailable"

    # Build knowledge context
    knowledge_context = "\n".join([
        f"- {k.get('content', '')[:200]}"
        for k in knowledge[:3]
    ]) if knowledge else ""

    prompt = f"""You are a certified financial planner giving serious, actionable advice.

User Profile:
- Country: {country.upper()}
- Monthly Income: {currency}{monthly_income:,}
- Current Inflation: {inflation_rate}%
- Budget Framework: {framework} rule

Current Spending:
{chr(10).join([f'- {k}: {currency}{v:,}' for k, v in spending.items()])}

If they invest their surplus monthly:
{proj_summary}

Tax Tip: {tax_tip}

Financial Knowledge Context:
{knowledge_context}

Write a Coach Plan (250-300 words) with these sections:
1. **Budget Rebuild** — show the {framework} breakdown with their actual income
2. **Top 3 Actions** — specific, numbered steps to take this week
3. **Investing Starter Plan** — where to start, how much, why
4. **The 10-Year Picture** — what disciplined investing looks like for them

Be specific with numbers. Cite the framework. Make it feel achievable."""

    response = model.generate_content(prompt)
    coach_plan = response.text

    # Build rebuilt budget
    if country == "india":
        rebuilt_budget = {
            "needs": round(monthly_income * 0.40, 2),
            "wants": round(monthly_income * 0.30, 2),
            "savings_investments": round(monthly_income * 0.30, 2),
            "framework": "40/30/30"
        }
    else:
        rebuilt_budget = {
            "needs": round(monthly_income * 0.50, 2),
            "wants": round(monthly_income * 0.30, 2),
            "savings_investments": round(monthly_income * 0.20, 2),
            "framework": "50/30/20"
        }

    steps = add_step(state, "Building your personalized coach plan", "Coach plan ready")

    return {
        **state,
        "coach_plan": coach_plan,
        "rebuilt_budget": rebuilt_budget,
        "steps": steps
    }