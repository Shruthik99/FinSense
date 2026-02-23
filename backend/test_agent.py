import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from agent.graph import run_agent

# Test budget — India, bad spending habits
test_budget = {
    "country": "india",
    "monthly_income": 50000,
    "language": "english",
    "spending": {
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
}

print("Running FinSense agent...")
print("=" * 60)

result = run_agent(test_budget)

print("\n📊 HEALTH SCORE:", result["health_score"], "— Grade:", result["health_grade"])
print("\n🔥 ROAST:")
print(result["roast"])
print("\n📈 COACH PLAN:")
print(result["coach_plan"])
print("\n💰 REBUILT BUDGET:")
print(result["rebuilt_budget"])
print("\n📰 AGENT STEPS:")
for step in result["steps"]:
    print(f"  Step {step['step_number']}: {step['step_name']} — {step['detail']}")