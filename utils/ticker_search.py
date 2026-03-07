import yfinance as yf
from fuzzywuzzy import fuzz

# Expanded ticker + company name database
TICKER_DATABASE = {
    # Large Cap Stocks
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
    "PG": "Procter & Gamble Co.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "The Home Depot Inc.",
    "MA": "Mastercard Inc.",
    "DIS": "The Walt Disney Company",
    "BAC": "Bank of America Corp.",
    "XOM": "Exxon Mobil Corporation",
    "PFE": "Pfizer Inc.",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo Inc.",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co. Inc.",
    "CVX": "Chevron Corporation",
    "LLY": "Eli Lilly and Company",
    "AVGO": "Broadcom Inc.",
    "COST": "Costco Wholesale Corporation",
    "TMO": "Thermo Fisher Scientific Inc.",
    "MCD": "McDonald's Corporation",
    "ACN": "Accenture plc",
    "NEE": "NextEra Energy Inc.",
    "TXN": "Texas Instruments Inc.",
    "PM": "Philip Morris International",
    "UPS": "United Parcel Service Inc.",
    "MS": "Morgan Stanley",
    "GS": "Goldman Sachs Group Inc.",
    "BLK": "BlackRock Inc.",
    "SCHW": "Charles Schwab Corporation",
    "AXP": "American Express Company",
    "SPGI": "S&P Global Inc.",
    "NFLX": "Netflix Inc.",
    "SPOT": "Spotify Technology S.A.",
    "UBER": "Uber Technologies Inc.",
    "LYFT": "Lyft Inc.",
    "ABNB": "Airbnb Inc.",
    "DASH": "DoorDash Inc.",
    "SHOP": "Shopify Inc.",
    "SQ": "Block Inc.",
    "PYPL": "PayPal Holdings Inc.",
    "ROKU": "Roku Inc.",
    "ZM": "Zoom Video Communications",
    "DOCU": "DocuSign Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "SNOW": "Snowflake Inc.",
    "COIN": "Coinbase Global Inc.",
    "RBLX": "Roblox Corporation",
    "HOOD": "Robinhood Markets Inc.",
    "SOFI": "SoFi Technologies Inc.",
    "RIVN": "Rivian Automotive Inc.",
    "NIO": "NIO Inc.",
    "BABA": "Alibaba Group Holding Ltd.",
    "AMD": "Advanced Micro Devices Inc.",
    "INTC": "Intel Corporation",
    "QCOM": "Qualcomm Inc.",
    "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corporation",
    "IBM": "International Business Machines",
    "CSCO": "Cisco Systems Inc.",
    "ADBE": "Adobe Inc.",
    "NOW": "ServiceNow Inc.",
    "INTU": "Intuit Inc.",
    "AMAT": "Applied Materials Inc.",
    "MU": "Micron Technology Inc.",
    "LRCX": "Lam Research Corporation",
    "KLAC": "KLA Corporation",
    "PANW": "Palo Alto Networks Inc.",
    "CRWD": "CrowdStrike Holdings Inc.",
    "ZS": "Zscaler Inc.",
    "NET": "Cloudflare Inc.",
    "DDOG": "Datadog Inc.",
    "MDB": "MongoDB Inc.",
    "TEAM": "Atlassian Corporation",
    "HUBS": "HubSpot Inc.",

    # ETFs
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (Nasdaq 100)",
    "IWM": "iShares Russell 2000 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "GLD": "SPDR Gold Shares ETF",
    "SLV": "iShares Silver Trust ETF",
    "USO": "United States Oil Fund ETF",
    "XLE": "Energy Select Sector SPDR ETF",
    "XLF": "Financial Select Sector SPDR ETF",
    "XLK": "Technology Select Sector SPDR ETF",
    "XLV": "Health Care Select Sector SPDR ETF",
    "ARKK": "ARK Innovation ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",

    # Bonds
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core US Aggregate Bond ETF",
    "LQD": "iShares iBoxx Investment Grade ETF",
    "HYG": "iShares iBoxx High Yield Bond ETF",
    "TIP": "iShares TIPS Bond ETF",

    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "XRP-USD": "XRP",
    "ADA-USD": "Cardano",
    "SOL-USD": "Solana",
    "DOGE-USD": "Dogecoin",
    "DOT-USD": "Polkadot",
    "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink",

    # Commodities
    "GC=F": "Gold Futures",
    "SI=F": "Silver Futures",
    "CL=F": "Crude Oil Futures",
    "NG=F": "Natural Gas Futures",
    "HG=F": "Copper Futures",
    "ZW=F": "Wheat Futures",
    "ZC=F": "Corn Futures",
    "ZS=F": "Soybean Futures",
}


def search_tickers(query: str, max_results: int = 8) -> list:
    """
    Searches both ticker symbols and company names using fuzzy matching.
    Returns list of (ticker, company_name) tuples.
    """
    if not query or len(query) < 1:
        return []

    query_upper = query.upper()
    query_lower = query.lower()
    scored = []

    for ticker, name in TICKER_DATABASE.items():
        ticker_score = fuzz.partial_ratio(query_upper, ticker)
        name_score   = fuzz.partial_ratio(query_lower, name.lower())
        best_score   = max(ticker_score, name_score)

        # Boost exact ticker prefix matches to the top
        if ticker.startswith(query_upper):
            best_score += 30

        scored.append((best_score, ticker, name))

    scored.sort(reverse=True)
    return [(ticker, name) for _, ticker, name in scored[:max_results]]