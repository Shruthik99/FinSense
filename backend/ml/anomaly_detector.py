import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ── Healthy Budget Benchmarks ────────────────────────────────────────────────
# These are the ideal percentage ranges for each spending category
# Based on established financial planning guidelines

INDIA_BENCHMARKS = {
    "rent":          {"ideal": 25, "warning": 35, "critical": 45},
    "food":          {"ideal": 10, "warning": 15, "critical": 20},
    "dining_out":    {"ideal": 5,  "warning": 10, "critical": 15},
    "transport":     {"ideal": 8,  "warning": 12, "critical": 18},
    "entertainment": {"ideal": 3,  "warning": 6,  "critical": 10},
    "subscriptions": {"ideal": 2,  "warning": 5,  "critical": 8},
    "shopping":      {"ideal": 5,  "warning": 10, "critical": 15},
    "health":        {"ideal": 3,  "warning": 6,  "critical": 10},
    "education":     {"ideal": 5,  "warning": 10, "critical": 15},
    "savings":       {"ideal": 20, "warning": 10, "critical": 5},
    "investments":   {"ideal": 10, "warning": 5,  "critical": 0},
    "other":         {"ideal": 5,  "warning": 8,  "critical": 12},
}

US_BENCHMARKS = {
    "rent":          {"ideal": 28, "warning": 35, "critical": 45},
    "food":          {"ideal": 8,  "warning": 12, "critical": 18},
    "dining_out":    {"ideal": 5,  "warning": 8,  "critical": 12},
    "transport":     {"ideal": 10, "warning": 15, "critical": 20},
    "entertainment": {"ideal": 3,  "warning": 6,  "critical": 10},
    "subscriptions": {"ideal": 3,  "warning": 6,  "critical": 10},
    "shopping":      {"ideal": 5,  "warning": 8,  "critical": 12},
    "health":        {"ideal": 5,  "warning": 8,  "critical": 12},
    "education":     {"ideal": 3,  "warning": 6,  "critical": 10},
    "savings":       {"ideal": 15, "warning": 8,  "critical": 3},
    "investments":   {"ideal": 10, "warning": 5,  "critical": 0},
    "other":         {"ideal": 5,  "warning": 8,  "critical": 12},
}


def detect_anomalies(spending: dict, monthly_income: float, country: str) -> list:
    """
    Detects anomalous spending categories using Isolation Forest.
    Compares user's spending percentages against healthy benchmarks.
    """
    benchmarks = INDIA_BENCHMARKS if country == "india" else US_BENCHMARKS
    currency = "₹" if country == "india" else "$"

    # Convert spending to percentages of income
    spending_percentages = {}
    for category, amount in spending.items():
        spending_percentages[category] = round((amount / monthly_income) * 100, 2)

    # Prepare data for Isolation Forest
    # We use deviation from benchmark as the feature
    deviations = []
    categories = []

    for category, percentage in spending_percentages.items():
        if category in benchmarks:
            ideal = benchmarks[category]["ideal"]
            # For savings/investments, negative deviation is bad
            if category in ["savings", "investments"]:
                deviation = ideal - percentage  # positive = underspending (bad)
            else:
                deviation = percentage - ideal  # positive = overspending (bad)
            deviations.append([deviation])
            categories.append(category)

    if not deviations:
        return []

    # Run Isolation Forest
    # contamination=0.3 means we expect ~30% of categories to be anomalous
    iso_forest = IsolationForest(
        contamination=0.3,
        random_state=42,
        n_estimators=100
    )
    predictions = iso_forest.fit_predict(deviations)
    scores = iso_forest.score_samples(deviations)

    # Build results
    results = []
    for i, category in enumerate(categories):
        amount = spending.get(category, 0)
        percentage = spending_percentages.get(category, 0)
        benchmark = benchmarks[category]
        is_anomalous = predictions[i] == -1  # -1 means anomaly in Isolation Forest

        # Determine verdict based on benchmark thresholds
        if category in ["savings", "investments"]:
            if percentage >= benchmark["ideal"]:
                verdict = "healthy"
            elif percentage >= benchmark["warning"]:
                verdict = "warning"
            else:
                verdict = "critical"
        else:
            if percentage <= benchmark["ideal"]:
                verdict = "healthy"
            elif percentage <= benchmark["warning"]:
                verdict = "warning"
            else:
                verdict = "critical"

        results.append({
            "category": category,
            "amount": amount,
            "percentage_of_income": percentage,
            "benchmark_percentage": benchmark["ideal"],
            "is_anomalous": is_anomalous,
            "anomaly_score": round(float(scores[i]), 4),
            "verdict": verdict,
            "currency": currency
        })

    # Sort — worst anomalies first
    results.sort(key=lambda x: (x["verdict"] == "critical", x["is_anomalous"]), reverse=True)

    return results


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with a bad budget
    test_spending = {
        "rent": 15000,
        "food": 5000,
        "dining_out": 8000,    # too high
        "transport": 3000,
        "entertainment": 4000, # too high
        "subscriptions": 3000, # too high
        "shopping": 6000,      # too high
        "health": 1000,
        "education": 1000,
        "savings": 2000,       # too low
        "investments": 0,      # critical
        "other": 2000
    }
    monthly_income = 50000

    print("Testing anomaly detection (India, ₹50,000 income)...")
    results = detect_anomalies(test_spending, monthly_income, "india")

    for r in results:
        status = "🔴" if r["verdict"] == "critical" else "🟡" if r["verdict"] == "warning" else "🟢"
        print(f"{status} {r['category']}: ₹{r['amount']:,} ({r['percentage_of_income']}% vs {r['benchmark_percentage']}% ideal) — {r['verdict']}")