from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def calculate_sip(
    monthly_amount: float,
    annual_return_rate: float,
    years: int,
    country: str = "india"
) -> dict:
    """
    Calculates SIP (Systematic Investment Plan) returns.
    Used for India — standard way Indians invest in mutual funds.
    Formula: FV = P × ((1 + r)^n - 1) / r × (1 + r)
    """
    try:
        monthly_rate = annual_return_rate / 100 / 12
        total_months = years * 12

        future_value = monthly_amount * (
            ((1 + monthly_rate) ** total_months - 1) / monthly_rate
        ) * (1 + monthly_rate)

        total_invested = monthly_amount * total_months
        total_returns = future_value - total_invested

        return {
            "type": "SIP",
            "monthly_amount": round(monthly_amount, 2),
            "years": years,
            "annual_return_rate": annual_return_rate,
            "future_value": round(future_value, 2),
            "total_invested": round(total_invested, 2),
            "total_returns": round(total_returns, 2),
            "return_percentage": round((total_returns / total_invested) * 100, 2),
            "currency": "INR" if country == "india" else "USD",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


def calculate_compound_interest(
    monthly_amount: float,
    annual_return_rate: float,
    years: int,
    country: str = "us"
) -> dict:
    """
    Calculates compound interest returns for monthly contributions.
    Used for US — standard way to show index fund / 401k growth.
    """
    try:
        monthly_rate = annual_return_rate / 100 / 12
        total_months = years * 12

        future_value = monthly_amount * (
            ((1 + monthly_rate) ** total_months - 1) / monthly_rate
        )

        total_invested = monthly_amount * total_months
        total_returns = future_value - total_invested

        return {
            "type": "Compound Interest",
            "monthly_amount": round(monthly_amount, 2),
            "years": years,
            "annual_return_rate": annual_return_rate,
            "future_value": round(future_value, 2),
            "total_invested": round(total_invested, 2),
            "total_returns": round(total_returns, 2),
            "return_percentage": round((total_returns / total_invested) * 100, 2),
            "currency": "INR" if country == "india" else "USD",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


def generate_projections(
    monthly_investable: float,
    country: str
) -> dict:
    """
    Generates investment projections for 5, 10, and 20 years.
    Uses realistic historical return rates:
    - India Nifty 50: ~12% CAGR historically
    - US S&P 500: ~10% CAGR historically
    """
    annual_rate = 12.0 if country == "india" else 10.0
    calculator = calculate_sip if country == "india" else calculate_compound_interest

    projections = {}
    for years in [5, 10, 20]:
        result = calculator(monthly_investable, annual_rate, years, country)
        projections[f"{years}_years"] = result

    return {
        "country": country,
        "monthly_investable_amount": monthly_investable,
        "assumed_annual_return": annual_rate,
        "benchmark": "Nifty 50 historical CAGR" if country == "india" else "S&P 500 historical CAGR",
        "projections": projections,
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }


def calculate_ppf_returns(
    annual_amount: float,
    years: int = 15
) -> dict:
    """
    Calculates PPF (Public Provident Fund) returns — India only.
    Current rate: 7.1% per annum (as of 2026)
    Lock-in period: 15 years minimum
    """
    ppf_rate = 7.1 / 100
    total = 0

    for year in range(1, years + 1):
        total = (total + annual_amount) * (1 + ppf_rate)

    total_invested = annual_amount * years
    total_returns = total - total_invested

    return {
        "type": "PPF",
        "annual_amount": round(annual_amount, 2),
        "years": years,
        "guaranteed_rate": 7.1,
        "maturity_amount": round(total, 2),
        "total_invested": round(total_invested, 2),
        "total_returns": round(total_returns, 2),
        "currency": "INR",
        "note": "PPF returns are tax-free under Section 80C",
        "status": "success"
    }


def calculate_hysa_returns(
    monthly_amount: float,
    years: int = 5
) -> dict:
    """
    Calculates High Yield Savings Account returns — US only.
    Current average HYSA rate: ~4.5% per annum (as of 2026)
    No lock-in period — fully liquid unlike PPF
    """
    try:
        annual_rate = 4.5 / 100
        monthly_rate = annual_rate / 12
        total_months = years * 12

        future_value = monthly_amount * (
            ((1 + monthly_rate) ** total_months - 1) / monthly_rate
        )

        total_invested = monthly_amount * total_months
        total_returns = future_value - total_invested

        return {
            "type": "HYSA",
            "monthly_amount": round(monthly_amount, 2),
            "years": years,
            "annual_rate": 4.5,
            "future_value": round(future_value, 2),
            "total_invested": round(total_invested, 2),
            "total_returns": round(total_returns, 2),
            "currency": "USD",
            "note": "HYSA is fully liquid — no lock-in period unlike PPF",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== INDIA ===")

    print("\nTesting SIP calculation...")
    sip = calculate_sip(5000, 12.0, 10, "india")
    print(f"₹5,000/month for 10 years at 12%:")
    print(f"  Invested: ₹{sip['total_invested']:,}")
    print(f"  Final value: ₹{sip['future_value']:,}")
    print(f"  Returns: ₹{sip['total_returns']:,} ({sip['return_percentage']}%)")

    print("\nTesting 5/10/20 year projections...")
    proj = generate_projections(10000, "india")
    for period, data in proj["projections"].items():
        print(f"  {period}: ₹{data['future_value']:,}")

    print("\nTesting PPF calculation...")
    ppf = calculate_ppf_returns(150000, 15)
    print(f"₹1,50,000/year for 15 years in PPF:")
    print(f"  Maturity: ₹{ppf['maturity_amount']:,}")

    print("\n=== US ===")

    print("\nTesting compound interest...")
    ci = calculate_compound_interest(500, 10.0, 10, "us")
    print(f"$500/month for 10 years at 10%:")
    print(f"  Invested: ${ci['total_invested']:,}")
    print(f"  Final value: ${ci['future_value']:,}")
    print(f"  Returns: ${ci['total_returns']:,} ({ci['return_percentage']}%)")

    print("\nTesting 5/10/20 year projections...")
    proj_us = generate_projections(500, "us")
    for period, data in proj_us["projections"].items():
        print(f"  {period}: ${data['future_value']:,}")

    print("\nTesting HYSA calculation...")
    hysa = calculate_hysa_returns(500, 5)
    print(f"$500/month for 5 years in HYSA at 4.5%:")
    print(f"  Invested: ${hysa['total_invested']:,}")
    print(f"  Maturity: ${hysa['future_value']:,}")