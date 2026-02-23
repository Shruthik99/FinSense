from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ── India Tax Slabs 2026 ─────────────────────────────────────────────────────

# New Tax Regime (default from FY 2024-25 onwards)
INDIA_NEW_TAX_SLABS = [
    (300000, 0),      # 0 - 3L: 0%
    (600000, 0.05),   # 3L - 6L: 5%
    (900000, 0.10),   # 6L - 9L: 10%
    (1200000, 0.15),  # 9L - 12L: 15%
    (1500000, 0.20),  # 12L - 15L: 20%
    (float('inf'), 0.30),  # Above 15L: 30%
]

# Old Tax Regime
INDIA_OLD_TAX_SLABS = [
    (250000, 0),      # 0 - 2.5L: 0%
    (500000, 0.05),   # 2.5L - 5L: 5%
    (1000000, 0.20),  # 5L - 10L: 20%
    (float('inf'), 0.30),  # Above 10L: 30%
]

# US Tax Brackets 2026 (Single filer)
US_TAX_BRACKETS = [
    (11925, 0.10),
    (48475, 0.12),
    (103350, 0.22),
    (197300, 0.24),
    (250525, 0.32),
    (626350, 0.35),
    (float('inf'), 0.37),
]


def calculate_india_tax(annual_income: float) -> dict:
    """
    Calculates India income tax under both old and new regimes.
    Recommends which regime is better for the user.
    """
    def compute_tax(income, slabs):
        tax = 0
        prev_limit = 0
        for limit, rate in slabs:
            if income <= prev_limit:
                break
            taxable = min(income, limit) - prev_limit
            tax += taxable * rate
            prev_limit = limit
        return tax

    new_regime_tax = compute_tax(annual_income, INDIA_NEW_TAX_SLABS)
    old_regime_tax = compute_tax(annual_income, INDIA_OLD_TAX_SLABS)

    # Section 80C max deduction
    max_80c = 150000

    # Old regime with 80C deduction
    old_regime_with_80c = compute_tax(
        max(0, annual_income - max_80c),
        INDIA_OLD_TAX_SLABS
    )

    # Add 4% health and education cess
    new_regime_tax *= 1.04
    old_regime_tax *= 1.04
    old_regime_with_80c *= 1.04

    # Which regime is better?
    better_regime = "new" if new_regime_tax <= old_regime_with_80c else "old"
    tax_savings_80c = old_regime_tax - old_regime_with_80c

    return {
        "country": "india",
        "annual_income": round(annual_income, 2),
        "new_regime_tax": round(new_regime_tax, 2),
        "old_regime_tax": round(old_regime_tax, 2),
        "old_regime_with_80c_tax": round(old_regime_with_80c, 2),
        "tax_savings_with_80c": round(tax_savings_80c, 2),
        "max_80c_deduction": max_80c,
        "recommended_regime": better_regime,
        "effective_rate_new": round((new_regime_tax / annual_income) * 100, 2),
        "effective_rate_old": round((old_regime_with_80c / annual_income) * 100, 2),
        "tip": f"Invest ₹1,50,000 in ELSS/PPF/NPS to save ₹{round(tax_savings_80c):,} in tax under old regime",
        "currency": "INR",
        "status": "success"
    }


def calculate_us_tax(annual_income: float) -> dict:
    """
    Calculates US federal income tax for single filer.
    Shows impact of 401k contributions on tax liability.
    """
    def compute_tax(income, brackets):
        tax = 0
        prev_limit = 0
        for limit, rate in brackets:
            if income <= prev_limit:
                break
            taxable = min(income, limit) - prev_limit
            tax += taxable * rate
            prev_limit = limit
        return tax

    # Standard deduction 2026
    standard_deduction = 14600
    taxable_income = max(0, annual_income - standard_deduction)
    base_tax = compute_tax(taxable_income, US_TAX_BRACKETS)

    # 401k max contribution 2026
    max_401k = 23000
    income_with_401k = max(0, taxable_income - max_401k)
    tax_with_401k = compute_tax(income_with_401k, US_TAX_BRACKETS)
    tax_savings_401k = base_tax - tax_with_401k

    return {
        "country": "us",
        "annual_income": round(annual_income, 2),
        "standard_deduction": standard_deduction,
        "taxable_income": round(taxable_income, 2),
        "federal_tax": round(base_tax, 2),
        "tax_with_max_401k": round(tax_with_401k, 2),
        "tax_savings_401k": round(tax_savings_401k, 2),
        "max_401k_contribution": max_401k,
        "effective_rate": round((base_tax / annual_income) * 100, 2),
        "tip": f"Contributing ${max_401k:,} to 401k saves you ${round(tax_savings_401k):,} in federal taxes",
        "currency": "USD",
        "status": "success"
    }


def get_tax_estimate(country: str, annual_income: float) -> dict:
    """
    Main function called by the MCP tool.
    Routes to the right function based on country.
    """
    if country == "india":
        return calculate_india_tax(annual_income)
    elif country == "us":
        return calculate_us_tax(annual_income)
    else:
        return {"error": f"Unknown country: {country}"}


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing India tax (₹8,00,000 annual income)...")
    india_tax = calculate_india_tax(800000)
    print(f"  New regime tax: ₹{india_tax['new_regime_tax']:,}")
    print(f"  Old regime tax: ₹{india_tax['old_regime_tax']:,}")
    print(f"  Old regime with 80C: ₹{india_tax['old_regime_with_80c_tax']:,}")
    print(f"  Recommended: {india_tax['recommended_regime']} regime")
    print(f"  Tip: {india_tax['tip']}")

    print("\nTesting US tax ($75,000 annual income)...")
    us_tax = calculate_us_tax(75000)
    print(f"  Federal tax: ${us_tax['federal_tax']:,}")
    print(f"  Tax with max 401k: ${us_tax['tax_with_max_401k']:,}")
    print(f"  Tax savings: ${us_tax['tax_savings_401k']:,}")
    print(f"  Tip: {us_tax['tip']}")