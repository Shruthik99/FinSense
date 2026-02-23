from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def compute_health_score(
    spending: dict,
    monthly_income: float,
    anomalies: list,
    country: str
) -> dict:
    """
    Computes a financial health score from 0 to 100.
    Based on 5 weighted factors:
    1. Savings rate (30 points)
    2. Investment rate (25 points)
    3. Anomaly penalty (20 points)
    4. Essential vs discretionary ratio (15 points)
    5. Income coverage (10 points)
    """

    score = 0
    breakdown = {}

    # ── Factor 1: Savings Rate (30 points) ──────────────────────────────────
    savings = spending.get("savings", 0)
    savings_rate = (savings / monthly_income) * 100
    ideal_savings = 20 if country == "india" else 15

    savings_score = min(30, (savings_rate / ideal_savings) * 30)
    score += savings_score
    breakdown["savings_rate"] = {
        "score": round(savings_score, 1),
        "max": 30,
        "your_rate": round(savings_rate, 1),
        "ideal_rate": ideal_savings,
        "verdict": "good" if savings_rate >= ideal_savings else "needs improvement"
    }

    # ── Factor 2: Investment Rate (25 points) ────────────────────────────────
    investments = spending.get("investments", 0)
    investment_rate = (investments / monthly_income) * 100
    ideal_investment = 10

    investment_score = min(25, (investment_rate / ideal_investment) * 25)
    score += investment_score
    breakdown["investment_rate"] = {
        "score": round(investment_score, 1),
        "max": 25,
        "your_rate": round(investment_rate, 1),
        "ideal_rate": ideal_investment,
        "verdict": "good" if investment_rate >= ideal_investment else "needs improvement"
    }

    # ── Factor 3: Anomaly Penalty (20 points) ────────────────────────────────
    critical_count = len([a for a in anomalies if a["verdict"] == "critical"])
    warning_count = len([a for a in anomalies if a["verdict"] == "warning"])

    anomaly_penalty = (critical_count * 4) + (warning_count * 2)
    anomaly_score = max(0, 20 - anomaly_penalty)
    score += anomaly_score
    breakdown["spending_health"] = {
        "score": round(anomaly_score, 1),
        "max": 20,
        "critical_issues": critical_count,
        "warnings": warning_count,
        "verdict": "good" if critical_count == 0 else "needs improvement"
    }

    # ── Factor 4: Essential vs Discretionary Ratio (15 points) ──────────────
    essential = (spending.get("rent", 0) + spending.get("food", 0) +
                 spending.get("transport", 0) + spending.get("health", 0))
    discretionary = (spending.get("dining_out", 0) + spending.get("entertainment", 0) +
                     spending.get("subscriptions", 0) + spending.get("shopping", 0))

    if discretionary > 0:
        ratio = essential / (essential + discretionary)
        ratio_score = min(15, ratio * 15)
    else:
        ratio_score = 15

    score += ratio_score
    breakdown["spending_ratio"] = {
        "score": round(ratio_score, 1),
        "max": 15,
        "essential_spend": round(essential, 2),
        "discretionary_spend": round(discretionary, 2),
        "verdict": "good" if ratio_score >= 10 else "needs improvement"
    }

    # ── Factor 5: Income Coverage (10 points) ────────────────────────────────
    total_spending = sum(spending.values())
    coverage_rate = (total_spending / monthly_income) * 100

    if coverage_rate <= 70:
        coverage_score = 10
    elif coverage_rate <= 85:
        coverage_score = 6
    elif coverage_rate <= 95:
        coverage_score = 3
    else:
        coverage_score = 0

    score += coverage_score
    breakdown["income_coverage"] = {
        "score": round(coverage_score, 1),
        "max": 10,
        "spending_percentage": round(coverage_rate, 1),
        "verdict": "good" if coverage_rate <= 70 else "needs improvement"
    }

    # ── Final Score ──────────────────────────────────────────────────────────
    final_score = round(min(100, score), 1)

    if final_score >= 80:
        grade = "A"
        summary = "Excellent financial health — keep it up!"
    elif final_score >= 65:
        grade = "B"
        summary = "Good financial health with room to improve"
    elif final_score >= 50:
        grade = "C"
        summary = "Average financial health — some areas need attention"
    elif final_score >= 35:
        grade = "D"
        summary = "Poor financial health — significant changes needed"
    else:
        grade = "F"
        summary = "Critical financial health — urgent action required"

    return {
        "score": final_score,
        "grade": grade,
        "summary": summary,
        "breakdown": breakdown,
        "status": "success"
    }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ml.anomaly_detector import detect_anomalies

    # ── India Test ───────────────────────────────────────────────────────────
    test_spending_india = {
        "rent": 15000,
        "food": 5000,
        "dining_out": 8000,
        "transport": 3000,
        "entertainment": 4000,
        "subscriptions": 3000,
        "shopping": 6000,
        "health": 1000,
        "education": 1000,
        "savings": 2000,
        "investments": 0,
        "other": 2000
    }
    monthly_income_india = 50000

    anomalies_india = detect_anomalies(test_spending_india, monthly_income_india, "india")
    result_india = compute_health_score(test_spending_india, monthly_income_india, anomalies_india, "india")

    print("=" * 50)
    print("INDIA TEST — ₹50,000 monthly income")
    print("=" * 50)
    print(f"Financial Health Score: {result_india['score']}/100 — Grade: {result_india['grade']}")
    print(f"Summary: {result_india['summary']}")
    print("\nBreakdown:")
    for factor, data in result_india["breakdown"].items():
        print(f"  {factor}: {data['score']}/{data['max']} — {data['verdict']}")

    # ── US Test ──────────────────────────────────────────────────────────────
    test_spending_us = {
        "rent": 1800,
        "food": 400,
        "dining_out": 600,
        "transport": 500,
        "entertainment": 300,
        "subscriptions": 200,
        "shopping": 400,
        "health": 200,
        "education": 100,
        "savings": 100,
        "investments": 0,
        "other": 150
    }
    monthly_income_us = 5000

    anomalies_us = detect_anomalies(test_spending_us, monthly_income_us, "us")
    result_us = compute_health_score(test_spending_us, monthly_income_us, anomalies_us, "us")

    print("\n" + "=" * 50)
    print("US TEST — $5,000 monthly income")
    print("=" * 50)
    print(f"Financial Health Score: {result_us['score']}/100 — Grade: {result_us['grade']}")
    print(f"Summary: {result_us['summary']}")
    print("\nBreakdown:")
    for factor, data in result_us["breakdown"].items():
        print(f"  {factor}: {data['score']}/{data['max']} — {data['verdict']}")