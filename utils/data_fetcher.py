import yfinance as yf
import pandas as pd 

def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical closing prices for a list of tickers.
    
    Parameters:
        tickers    : list of ticker symbols e.g. ["AAPL", "BTC-USD", "GLD"]
        start_date : string in "YYYY-MM-DD" format
        end_date   : string in "YYYY-MM-DD" format
    
    Returns:
        A DataFrame where each column is a ticker and each row is a date
    """
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    # Extract only the closing prices
    price_data = raw_data["Close"]
    
    # Drop any dates where data is missing
    price_data = price_data.dropna()
    
    return price_data


def fetch_asset_info(ticker: str) -> dict:
    """
    Fetches basic info about an asset (name, sector, currency).
    
    Parameters:
        ticker : a single ticker symbol e.g. "AAPL"
    
    Returns:
        A dictionary with basic asset metadata
    """
    asset = yf.Ticker(ticker)
    info = asset.info
    
    return {
        "name"    : info.get("longName", ticker),
        "sector"  : info.get("sector", "N/A"),
        "currency": info.get("currency", "USD")
    }