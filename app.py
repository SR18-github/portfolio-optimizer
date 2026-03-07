import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data_fetcher import fetch_price_data
from utils.optimizer import optimize_portfolio
from utils.ticker_search import search_tickers
from utils.analytics import (
    calculate_risk_metrics,
    calculate_correlation,
    forecast_prices,
    monte_carlo_simulation,
    rebalancing_suggestions
)

# --- Page Config ---
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Mean Variance Portfolio Optimizer")
st.markdown("Search and add assets by name or ticker, then optimize your portfolio.")

# --- Session State ---
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = []
if "result" not in st.session_state:
    st.session_state.result = None
if "price_data" not in st.session_state:
    st.session_state.price_data = None

# --- Search Section ---
st.subheader("🔍 Search & Add Assets")

search_query = st.text_input(
    "Search by ticker or company name",
    placeholder="e.g. Apple, Bitcoin, Gold, SPY...",
    label_visibility="collapsed"
)

if search_query:
    suggestions = search_tickers(search_query)
    if suggestions:
        st.markdown("**Click an asset to add it:**")
        cols = st.columns(2)
        for i, (ticker, name) in enumerate(suggestions):
            with cols[i % 2]:
                already_added = ticker in st.session_state.selected_tickers
                label = f"✅ {ticker} — {name}" if already_added else f"➕ {ticker} — {name}"
                if st.button(label, key=f"add_{ticker}", use_container_width=True):
                    if not already_added:
                        st.session_state.selected_tickers.append(ticker)
                        st.rerun()
    else:
        st.caption("No matches found. Try a different name or ticker symbol.")

st.divider()

# --- Selected Assets ---
st.subheader("📋 Your Portfolio Assets")

if st.session_state.selected_tickers:
    for ticker in st.session_state.selected_tickers:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{ticker}**")
        with col2:
            if st.button("Remove", key=f"remove_{ticker}"):
                st.session_state.selected_tickers.remove(ticker)
                st.rerun()
else:
    st.info("No assets added yet. Use the search bar above to build your portfolio.")

st.divider()

# --- Settings ---
st.subheader("⚙️ Settings")
col1, col2, col3, col4 = st.columns(4)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
with col3:
    num_portfolios = st.slider("Simulated Portfolios", 1000, 10000, 5000, 500)
with col4:
    initial_investment = st.number_input(
        "Initial Investment ($)", min_value=1000, value=10000, step=1000
    )

st.divider()

# --- Optimize Button ---
run_button = st.button(
    "🚀 Optimize Portfolio",
    use_container_width=True,
    disabled=len(st.session_state.selected_tickers) < 2
)

if len(st.session_state.selected_tickers) < 2:
    st.caption("⚠️ Add at least 2 assets to run optimization.")

if run_button:
    with st.spinner("Fetching data and optimizing..."):
        try:
            price_data = fetch_price_data(
                st.session_state.selected_tickers,
                str(start_date),
                str(end_date)
            )
            result = optimize_portfolio(price_data, num_portfolios=num_portfolios)
            st.session_state.result = result
            st.session_state.price_data = price_data
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Double-check your ticker symbols and date range.")

# --- Results ---
if st.session_state.result:
    result     = st.session_state.result
    price_data = st.session_state.price_data

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Allocation",
        "⚠️ Risk Scores",
        "🔥 Correlation",
        "📅 Forecast",
        "🎲 Monte Carlo",
        "⚖️ Rebalancing"
    ])

    # ── Tab 1: Allocation ─────────────────────────────────────────────────────
    with tab1:
        st.subheader("Optimal Portfolio Allocation")
        weights_df = pd.DataFrame({
            "Ticker": result["tickers"],
            "Weight": result["optimal_weights"]
        })
        weights_df["Weight %"] = (weights_df["Weight"] * 100).round(2)

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(
                weights_df, names="Ticker", values="Weight",
                title="Portfolio Allocation"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.metric("📈 Expected Annual Return",
                      f"{result['optimal_return']*100:.2f}%")
            st.metric("📉 Annual Volatility",
                      f"{result['optimal_volatility']*100:.2f}%")
            st.metric("⚖️ Sharpe Ratio",
                      f"{result['optimal_sharpe']:.4f}")
            st.dataframe(weights_df[["Ticker", "Weight %"]],
                         use_container_width=True)

        st.subheader("🎯 Efficient Frontier")
        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(
            x=result["all_volatilities"] * 100,
            y=result["all_returns"] * 100,
            mode="markers",
            marker=dict(
                color=result["all_sharpes"], colorscale="Viridis",
                showscale=True, size=4,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name="Simulated Portfolios"
        ))
        fig_frontier.add_trace(go.Scatter(
            x=[result["optimal_volatility"] * 100],
            y=[result["optimal_return"] * 100],
            mode="markers",
            marker=dict(color="red", size=14, symbol="star"),
            name="Optimal Portfolio"
        ))
        fig_frontier.update_layout(
            xaxis_title="Volatility / Risk (%)",
            yaxis_title="Expected Return (%)"
        )
        st.plotly_chart(fig_frontier, use_container_width=True)

    st.subheader("📊 Historical Normalized Prices")
try:
    if st.session_state.price_data is None or st.session_state.price_data.empty or len(st.session_state.price_data) < 2:
        st.warning("Not enough price data to display. Try a wider date range.")
    else:
        normalized = (st.session_state.price_data / st.session_state.price_data.iloc[0]) * 100
        fig_prices = px.line(normalized, title="Normalized Price History")
        fig_prices.update_layout(yaxis_title="Value (Starting = $100)")
        st.plotly_chart(fig_prices, use_container_width=True)
except Exception as e:
    st.warning("Could not display normalized prices. Try a different date range.")

    # ── Tab 2: Risk Scores ────────────────────────────────────────────────────
    with tab2:
        st.subheader("⚠️ Risk Scoring Per Asset")
        st.markdown("""
        - **Annual Volatility** — how much the price fluctuates per year
        - **Max Drawdown** — the worst peak-to-trough loss in the period
        - **VaR 95%** — on a bad day, you'd expect to lose at least this much
        - **Risk Score** — 1 (very safe) to 10 (very risky)
        """)

        risk_df = calculate_risk_metrics(price_data)
        st.dataframe(risk_df, use_container_width=True)

        # Risk score bar chart
        risk_scores = calculate_risk_metrics(price_data).reset_index()
        fig_risk = px.bar(
            risk_scores, x="Ticker", y="Risk Score",
            color="Risk Score", color_continuous_scale="RdYlGn_r",
            title="Risk Score by Asset (1 = Safe, 10 = Risky)"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Tab 3: Correlation ────────────────────────────────────────────────────
    with tab3:
        st.subheader("🔥 Correlation Heatmap")
        st.markdown("""
        Shows how closely assets move together.
        - **+1.0** = move in perfect lockstep
        - **0.0** = no relationship
        - **-1.0** = move in opposite directions (great for diversification!)
        """)

        corr_matrix = calculate_correlation(price_data)
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Asset Correlation Matrix"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Tab 4: Forecast ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("📅 Price Forecasting (ARIMA)")
        st.markdown("Forecasts are based on historical patterns. Not financial advice!")

        forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30)

        with st.spinner("Running forecasts..."):
            forecasts = forecast_prices(price_data, days=forecast_days)

        for ticker, forecast in forecasts.items():
            if forecast is not None:
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data[ticker],
                    name="Historical",
                    line=dict(color="blue")
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast.index,
                    y=forecast.values,
                    name="Forecast",
                    line=dict(color="orange", dash="dash")
                ))
                fig_forecast.update_layout(
                    title=f"{ticker} — Price Forecast ({forecast_days} days)",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

    # ── Tab 5: Monte Carlo ────────────────────────────────────────────────────
    with tab5:
        st.subheader("🎲 Monte Carlo Future Simulation")
        st.markdown(f"""
        Simulates **1,000 possible futures** for your optimized portfolio
        over the next trading year based on historical return patterns.
        """)

        mc = monte_carlo_simulation(
            price_data,
            result["optimal_weights"],
            initial_investment=initial_investment
        )

        # Plot subset of simulation paths
        fig_mc = go.Figure()
        for i in range(0, 1000, 10):  # plot every 10th path to keep it clean
            fig_mc.add_trace(go.Scatter(
                y=mc["simulations"][:, i],
                mode="lines",
                line=dict(width=0.5, color="lightblue"),
                showlegend=False,
                opacity=0.3
            ))

        # Add median path
        median_path = np.median(mc["simulations"], axis=1)
        fig_mc.add_trace(go.Scatter(
            y=median_path,
            mode="lines",
            line=dict(color="blue", width=2),
            name="Median"
        ))

        fig_mc.update_layout(
            title="Monte Carlo Portfolio Simulation (1 Year)",
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)"
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Initial Investment",
                    f"${initial_investment:,.0f}")
        col2.metric("📈 Median Outcome",
                    f"${mc['median_final']:,.0f}")
        col3.metric("🐻 Worst 5% Case",
                    f"${mc['percentile_5']:,.0f}")
        col4.metric("🚀 Best 95% Case",
                    f"${mc['percentile_95']:,.0f}")

    # ── Tab 6: Rebalancing ────────────────────────────────────────────────────
    with tab6:
        st.subheader("⚖️ Rebalancing Suggestions")
        st.markdown("""
        Compares an **equal-weighted** starting portfolio against the
        **optimal weights** and tells you exactly what to buy or sell.
        """)

        rebal_df = rebalancing_suggestions(
            result["tickers"],
            result["optimal_weights"],
            portfolio_value=initial_investment
        )
        st.dataframe(rebal_df, use_container_width=True)

        fig_rebal = go.Figure()
        fig_rebal.add_trace(go.Bar(
            name="Current (Equal Weight)",
            x=rebal_df.index,
            y=rebal_df["Current Weight"]
        ))
        fig_rebal.add_trace(go.Bar(
            name="Optimal Weight",
            x=rebal_df.index,
            y=rebal_df["Optimal Weight"]
        ))
        fig_rebal.update_layout(
            barmode="group",
            title="Current vs Optimal Weights (%)",
            yaxis_title="Weight (%)"
        )
        st.plotly_chart(fig_rebal, use_container_width=True)