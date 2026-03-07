import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts daily prices into daily percentage returns.
    e.g. if AAPL goes from $100 to $102, that's a 2% return.
    """
    return price_data.pct_change().dropna()


def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=252):
    """
    Given a set of weights, calculates the portfolio's:
        - Annualized return
        - Annualized volatility (risk)
        - Sharpe Ratio (return per unit of risk)
    """
    # Annualized return
    annual_return = np.sum(mean_returns * weights) * trading_days

    # Annualized volatility
    annual_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix * trading_days, weights))
    )

    # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = annual_return / annual_volatility

    return annual_return, annual_volatility, sharpe_ratio


def minimize_volatility(weights, mean_returns, cov_matrix):
    """Helper used by the optimizer — returns only volatility to minimize."""
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def optimize_portfolio(price_data: pd.DataFrame, num_portfolios=5000):
    """
    Core function — runs two things:
        1. Monte Carlo Simulation: generates thousands of random portfolios
        2. Optimization: finds the single best (max Sharpe) portfolio
    
    Returns a dictionary with everything needed for the UI.
    """
    returns = calculate_returns(price_data)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(price_data.columns)
    tickers = list(price_data.columns)

    # --- Monte Carlo Simulation ---
    results = np.zeros((3, num_portfolios))  # rows: return, volatility, sharpe
    all_weights = []

    for i in range(num_portfolios):
        # Random weights that sum to 1
        w = np.random.random(num_assets)
        w /= np.sum(w)
        all_weights.append(w)

        ret, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe

    # --- Find Max Sharpe Portfolio via Optimization ---
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # weights must sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))              # no short selling
    initial_weights = [1 / num_assets] * num_assets                # start equal-weighted

    result = minimize(
        minimize_volatility,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    opt_return, opt_volatility, opt_sharpe = portfolio_performance(
        optimal_weights, mean_returns, cov_matrix
    )

    return {
        "tickers"         : tickers,
        "optimal_weights" : optimal_weights,
        "optimal_return"  : opt_return,
        "optimal_volatility": opt_volatility,
        "optimal_sharpe"  : opt_sharpe,
        "all_returns"     : results[0],
        "all_volatilities": results[1],
        "all_sharpes"     : results[2],
        "all_weights"     : all_weights
    }