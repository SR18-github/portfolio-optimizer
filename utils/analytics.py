import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


# ── 1. RISK SCORING ──────────────────────────────────────────────────────────

def calculate_risk_metrics(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each asset calculates:
        - Annualized Volatility
        - Max Drawdown (worst peak-to-trough loss)
        - Value at Risk 95% (worst expected daily loss 95% of the time)
        - Risk Score (1-10 scale)
    """
    returns = price_data.pct_change().dropna()
    metrics = []

    for ticker in price_data.columns:
        r = returns[ticker]

        # Annualized volatility
        volatility = r.std() * np.sqrt(252)

        # Max drawdown
        cumulative = (1 + r).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Value at Risk (95%)
        var_95 = np.percentile(r, 5)

        # Risk score 1-10 based on volatility
        risk_score = min(10, max(1, round(volatility * 20)))

        metrics.append({
            "Ticker"          : ticker,
            "Annual Volatility": f"{volatility*100:.1f}%",
            "Max Drawdown"    : f"{max_drawdown*100:.1f}%",
            "VaR 95%"         : f"{var_95*100:.2f}%",
            "Risk Score"      : risk_score
        })

    return pd.DataFrame(metrics).set_index("Ticker")


# ── 2. CORRELATION HEATMAP ────────────────────────────────────────────────────

def calculate_correlation(price_data: pd.DataFrame) -> pd.DataFrame:
    """Returns a correlation matrix of daily returns between all assets."""
    returns = price_data.pct_change().dropna()
    return returns.corr().round(2)


# ── 3. PRICE FORECASTING ──────────────────────────────────────────────────────

def forecast_prices(price_data: pd.DataFrame, days: int = 30) -> dict:
    """
    Forecasts future prices for each asset using ARIMA.
    Returns a dict of {ticker: forecast_series}
    """
    forecasts = {}

    for ticker in price_data.columns:
        try:
            series = price_data[ticker].dropna()

            # Fit ARIMA model
            model = ARIMA(series, order=(5, 1, 0))
            fitted = model.fit()

            # Forecast
            forecast = fitted.forecast(steps=days)
            last_date = series.index[-1]
            future_dates = pd.date_range(start=last_date, periods=days + 1, freq="B")[1:]
            forecasts[ticker] = pd.Series(forecast.values, index=future_dates)

        except Exception:
            forecasts[ticker] = None

    return forecasts


# ── 4. MONTE CARLO FUTURE SIMULATION ─────────────────────────────────────────

def monte_carlo_simulation(
    price_data: pd.DataFrame,
    weights: np.ndarray,
    days: int = 252,
    simulations: int = 1000,
    initial_investment: float = 10000
) -> dict:
    """
    Simulates 1000 possible futures for the optimized portfolio.
    Returns simulation paths and summary statistics.
    """
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    portfolio_simulations = np.zeros((days, simulations))

    for sim in range(simulations):
        # Generate random daily returns using multivariate normal distribution
        daily_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, days
        )
        # Apply weights to get portfolio daily return
        portfolio_daily = daily_returns.dot(weights)
        # Compound returns
        portfolio_simulations[:, sim] = (
            initial_investment * (1 + portfolio_daily).cumprod()
        )

    final_values = portfolio_simulations[-1, :]

    return {
        "simulations"    : portfolio_simulations,
        "mean_final"     : np.mean(final_values),
        "median_final"   : np.median(final_values),
        "percentile_5"   : np.percentile(final_values, 5),
        "percentile_95"  : np.percentile(final_values, 95),
        "initial"        : initial_investment,
        "days"           : days
    }


# ── 5. REBALANCING SUGGESTIONS ────────────────────────────────────────────────

def rebalancing_suggestions(
    tickers: list,
    optimal_weights: np.ndarray,
    current_weights: list = None,
    portfolio_value: float = 10000
) -> pd.DataFrame:
    """
    Compares current vs optimal weights and suggests trades.
    If no current weights provided, assumes equal weighting.
    """
    n = len(tickers)

    if current_weights is None:
        current_weights = [1 / n] * n

    current_weights = np.array(current_weights)
    diff = optimal_weights - current_weights

    df = pd.DataFrame({
        "Ticker"          : tickers,
        "Current Weight"  : (current_weights * 100).round(1),
        "Optimal Weight"  : (optimal_weights * 100).round(1),
        "Difference"      : (diff * 100).round(1),
        "Action ($)"      : (diff * portfolio_value).round(2)
    }).set_index("Ticker")

    df["Action"] = df["Difference"].apply(
        lambda x: "🟢 Buy" if x > 1 else ("🔴 Sell" if x < -1 else "✅ Hold")
    )

    return df