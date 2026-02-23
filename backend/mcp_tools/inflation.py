import requests
import os
from fredapi import Fred
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

# FRED series ID for US CPI (Consumer Price Index for All Urban Consumers)
US_CPI_SERIES = "CPIAUCSL"

# MOSPI API endpoint for India CPI data (no key needed — public government data)
INDIA_CPI_URL = "https://api.data.gov.in/resource/b4e6d07a-af0c-4f8e-9e9e-5f5a0a2a5a2a"


def get_us_inflation() -> dict:
    """
    Fetches the current US CPI inflation rate from FRED API.
    FRED is the Federal Reserve's official data source.
    """
    try:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))

        # Fetch the last 14 months of CPI data to ensure we have 12 months gap
        cpi_data = fred.get_series(US_CPI_SERIES)

        # Get the two most recent values that are 12 months apart
        # FRED returns a pandas Series with date index
        cpi_data = cpi_data.dropna()  # remove any null values

        # Most recent CPI value
        current_cpi = cpi_data.iloc[-1]

        # CPI value from exactly 12 months ago
        cpi_year_ago = cpi_data.iloc[-13]

        # Year over year inflation rate formula
        inflation_rate = round(((current_cpi - cpi_year_ago) / cpi_year_ago) * 100, 2)

        # Get the date of the most recent reading
        latest_date = cpi_data.index[-1].strftime("%B %Y")

        return {
            "country": "us",
            "inflation_rate": float(inflation_rate),
            "current_cpi_index": round(float(current_cpi), 3),
            "period": "Year-over-Year",
            "latest_reading": latest_date,
            "source": "US Federal Reserve (FRED)",
            "series": US_CPI_SERIES,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        return {
            "country": "us",
            "inflation_rate": 3.1,
            "source": "Fallback estimate (FRED unavailable)",
            "error": str(e),
            "status": "fallback"
        }


def get_india_inflation() -> dict:
    """
    Fetches India's current CPI inflation rate.
    Uses RBI's publicly available data — no API key needed.
    """
    try:
        # RBI publishes CPI data publicly
        # We fetch the latest available inflation figure
        url = "https://api.rbi.org.in/api/v1/inflation"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            inflation_rate = data.get("cpi_inflation", None)

            if inflation_rate:
                return {
                    "country": "india",
                    "inflation_rate": round(float(inflation_rate), 2),
                    "source": "Reserve Bank of India (RBI)",
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }

        # If RBI API doesn't return data, use World Bank API as backup
        raise Exception("RBI API returned no data")

    except Exception:
        try:
            # World Bank API for India inflation — free, no key
            url = "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL.ZG?format=json&mrv=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            inflation_rate = data[1][0]["value"]

            if inflation_rate:
                return {
                    "country": "india",
                    "inflation_rate": round(float(inflation_rate), 2),
                    "source": "World Bank (India CPI)",
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
        except Exception as e:
            pass

        # Final fallback
        return {
            "country": "india",
            "inflation_rate": 4.8,
            "source": "Fallback estimate",
            "status": "fallback"
        }


def get_inflation(country: str) -> dict:
    """
    Main function called by the MCP tool.
    Routes to the right function based on country.
    """
    if country == "india":
        return get_india_inflation()
    elif country == "us":
        return get_us_inflation()
    else:
        return {"error": f"Unknown country: {country}"}


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing India inflation...")
    print(get_india_inflation())

    print("\nTesting US inflation...")
    print(get_us_inflation())