from pydantic import BaseModel, Field
from typing import Optional

# ── What the frontend sends to the backend ──────────────────────────────────

class SpendingCategories(BaseModel):
    """
    Breakdown of monthly spending across categories.
    All values are in the user's local currency (INR or USD).
    """
    rent: float = Field(ge=0, description="Monthly rent or housing cost")
    food: float = Field(ge=0, description="Groceries and home cooking")
    dining_out: float = Field(ge=0, description="Restaurants, Swiggy, Zomato, DoorDash")
    transport: float = Field(ge=0, description="Transport, fuel, Uber, metro")
    entertainment: float = Field(ge=0, description="Movies, events, hobbies")
    subscriptions: float = Field(ge=0, description="Netflix, Spotify, apps")
    shopping: float = Field(ge=0, description="Clothes, gadgets, misc shopping")
    health: float = Field(ge=0, description="Medical, gym, pharmacy")
    education: float = Field(ge=0, description="Courses, books, fees")
    savings: float = Field(ge=0, description="Amount currently being saved")
    investments: float = Field(ge=0, description="Amount currently being invested")
    other: float = Field(ge=0, description="Any other expenses")


class BudgetInput(BaseModel):
    """
    User Budget Submission Payload

    Represents the complete financial data entered by the user.
    This structured object is transmitted to the backend API
    when the user submits their budget for AI analysis.
    """
    country: str = Field(description="Either 'india' or 'us'")
    monthly_income: float = Field(gt=0, description="Total monthly take-home income")
    spending: SpendingCategories
    language: Optional[str] = Field(default="english", description="'english' or 'hinglish' — India only")


# ── What the backend sends back to the frontend ─────────────────────────────

class AnomalyResult(BaseModel):
    """
    Result of ML anomaly detection on a spending category.
    """
    category: str
    amount: float
    percentage_of_income: float
    is_anomalous: bool
    anomaly_score: float
    benchmark_percentage: float
    verdict: str  # "healthy", "warning", "critical"


class InvestmentRecommendation(BaseModel):
    """
    A single investment recommendation with live price data.
    """
    name: str
    ticker: Optional[str]
    type: str  # "index_fund", "mutual_fund", "ppf", "fd", "etf"
    suggested_monthly_amount: float
    current_price: Optional[float]
    expected_annual_return: float
    reason: str


class RebuiltBudget(BaseModel):
    """
    AI-Optimized Budget Allocation

Returns a reconstructed budget aligned with country-specific
financial best-practice frameworks:
- India: 40/30/30 allocation model
- United States: 50/30/20 allocation model

   The allocation is dynamically selected based on the user's location.
   """
    needs: float
    wants: float
    savings_and_investments: float
    framework_used: str  # "40/30/30" or "50/30/20"
    breakdown: dict  


class HealthScore(BaseModel):
    """
    Overall financial health score and its breakdown.
    """
    score: float  # 0 to 100
    grade: str    # "A", "B", "C", "D", "F"
    breakdown: dict  # what contributed to the score
    summary: str  # one sentence plain English explanation


class FinSenseResponse(BaseModel):

    """
    Comprehensive Budget Analysis Response

Encapsulates the full AI-generated output returned to the frontend,
including:
- Financial health score
- Behavioral feedback analysis
- Actionable improvement plan
- Personalized budget recommendations

   This object powers the post-analysis user dashboard experience.
   """
    roast: str
    coach_plan: str
    rebuilt_budget: RebuiltBudget
    anomalies: list[AnomalyResult]
    investment_recommendations: list[InvestmentRecommendation]
    health_score: HealthScore
    inflation_rate: float
    market_snapshot: dict  # live market data fetched during analysis
    top_news: list[dict]   # financial headlines
    sources_used: list[str]  # RAG sources cited in the advice


# ── Streaming response for the live agent reasoning log ─────────────────────

class AgentStep(BaseModel):
    """
    A single step in the agent's reasoning process.
    Streamed live to the frontend so the user can watch the agent think.
    """
    step_number: int
    step_name: str   # e.g. "Fetching inflation rate"
    status: str      # "running", "complete", "error"
    detail: Optional[str] = None  # optional extra info


# ── Chat interface ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """
    A message in the follow-up chat interface.
    """
    message: str
    country: str
    context: Optional[dict] = None  # the user's budget context for personalized answers