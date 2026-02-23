import yfinance as yf
import requests
from datetime import datetime

# ── Indian Market Tickers (NSE/BSE via yfinance) ─────────────────────────────
INDIA_TICKERS = {
    "Nifty 50":       "^NSEI",
    "Sensex":         "^BSESN",
    "Nifty Bank":     "^NSEBANK",
}

# ── US Market Tickers ────────────────────────────────────────────────────────
US_TICKERS = {
    "S&P 500":        "^GSPC",
    "Nasdaq":         "^IXIC",
    "VOO (S&P ETF)":  "VOO",
    "QQQ (Nasdaq ETF)": "QQQ",
    "VTI (Total Market)": "VTI",
}

# ── Top Indian Mutual Funds via MFAPI ────────────────────────────────────────
INDIA_FUNDS = {
    "Mirae Asset Large Cap Fund":     "119551",
    "Axis Bluechip Fund":             "120503",
    "Parag Parikh Flexi Cap Fund":    "122639",
    "HDFC Index Fund Nifty 50":       "120716",
}


def get_india_market_data() -> dict:
    """
    Fetches live Indian market data.
    Returns Nifty 50, Sensex prices + top mutual fund NAVs.
    """
    result = {
        "country": "india",
        "timestamp": datetime.now().isoformat(),
        "indices": {},
        "mutual_funds": {},
        "status": "success"
    }

    # Fetch index prices via yfinance
    for name, ticker in INDIA_TICKERS.items():
        try:
            data = yf.Ticker(ticker)
            info = data.fast_info
            result["indices"][name] = {
                "price": round(info.last_price, 2),
                "currency": "INR"
            }
        except Exception as e:
            result["indices"][name] = {"error": str(e)}

    # Fetch mutual fund NAVs via MFAPI (free, no key needed)
    for fund_name, scheme_code in INDIA_FUNDS.items():
        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(url, timeout=5)
            data = response.json()
            latest_nav = data["data"][0]  # most recent NAV
            result["mutual_funds"][fund_name] = {
                "nav": float(latest_nav["nav"]),
                "date": latest_nav["date"],
                "currency": "INR"
            }
        except Exception as e:
            result["mutual_funds"][fund_name] = {"error": str(e)}

    return result


def get_us_market_data() -> dict:
    """
    Fetches live US market data.
    Returns S&P 500, Nasdaq prices + popular ETF prices.
    """
    result = {
        "country": "us",
        "timestamp": datetime.now().isoformat(),
        "indices": {},
        "etfs": {},
        "status": "success"
    }

    # Fetch all US tickers via yfinance
    for name, ticker in US_TICKERS.items():
        try:
            data = yf.Ticker(ticker)
            info = data.fast_info
            result["indices" if "^" in ticker else "etfs"][name] = {
                "price": round(info.last_price, 2),
                "currency": "USD"
            }
        except Exception as e:
            result["indices" if "^" in ticker else "etfs"][name] = {
                "error": str(e)
            }

    return result


def get_market_data(country: str) -> dict:
    """
    Main function called by the MCP tool.
    Routes to the right function based on country.
    """
    if country == "india":
        return get_india_market_data()
    elif country == "us":
        return get_us_market_data()
    else:
        return {"error": f"Unknown country: {country}"}


# ── Quick test — run this file directly to verify it works ──────────────────
if __name__ == "__main__":
    print("Testing India market data...")
    india_data = get_india_market_data()
    print(india_data)

    print("\nTesting US market data...")
    us_data = get_us_market_data()
    print(us_data)