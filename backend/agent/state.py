from typing import TypedDict, Optional
from datetime import datetime


class AgentState(TypedDict):
    """
    The complete state of the agent at any point during reasoning.
    Think of this as the agent's working memory — everything it
    knows and has fetched lives here.
    """

    # ── User Input ───────────────────────────────────────────────────────────
    country: str                    # "india" or "us"
    monthly_income: float           # user's monthly income
    spending: dict                  # spending categories and amounts
    language: str                   # "english" or "hinglish"

    # ── ML Results ───────────────────────────────────────────────────────────
    anomalies: list                 # flagged spending anomalies
    health_score: float             # 0-100 financial health score
    health_grade: str               # A, B, C, D, F

    # ── Live Data (fetched by MCP tools) ─────────────────────────────────────
    inflation_data: dict            # current inflation rate
    market_data: dict               # live market prices
    news_articles: list             # financial headlines
    tax_estimate: dict              # tax calculation results

    # ── Investment Projections ───────────────────────────────────────────────
    projections: dict               # 5/10/20 year projections
    investment_recommendations: list # personalized investment suggestions

    # ── RAG Retrieved Knowledge ──────────────────────────────────────────────
    retrieved_knowledge: list       # relevant passages from knowledge base

    # ── Generated Outputs ────────────────────────────────────────────────────
    roast: str                      # the brutal roast text
    coach_plan: str                 # the serious coach plan
    rebuilt_budget: dict            # the rebuilt budget breakdown

    # ── Agent Reasoning Steps (streamed to frontend) ─────────────────────────
    steps: list                     # list of AgentStep dicts

    # ── Metadata ────────────────────────────────────────────────────────────
    timestamp: str                  # when analysis started
    error: Optional[str]            # any error that occurred


def create_initial_state(budget_input: dict) -> AgentState:
    """
    Creates the initial state from the user's budget input.
    All fields start empty — the agent fills them in as it reasons.
    """
    return AgentState(
        # From user input
        country=budget_input.get("country", "india"),
        monthly_income=budget_input.get("monthly_income", 0),
        spending=budget_input.get("spending", {}),
        language=budget_input.get("language", "english"),

        # Empty — filled by ML layer
        anomalies=[],
        health_score=0.0,
        health_grade="",

        # Empty — filled by MCP tools
        inflation_data={},
        market_data={},
        news_articles=[],
        tax_estimate={},

        # Empty — filled by calculator
        projections={},
        investment_recommendations=[],

        # Empty — filled by RAG
        retrieved_knowledge=[],

        # Empty — filled by LLM generation
        roast="",
        coach_plan="",
        rebuilt_budget={},

        # Empty — filled as agent progresses
        steps=[],

        # Metadata
        timestamp=datetime.now().isoformat(),
        error=None
    )