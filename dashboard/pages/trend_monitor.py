"""
Trend Monitor Page
Gas concentration and health index trend visualization with forecasting
"""

# Import dashboard components
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

dashboard_path = Path(__file__).parent.parent
sys.path.insert(0, str(dashboard_path))

from components.gas_chart import (
    create_forecast_chart,
    create_gas_trend_chart,
    create_multi_gas_chart,
)
from components.health_gauge import create_health_gauge

# API Configuration
API_BASE_URL = "http://localhost:8000"


@st.cache_data(ttl=60)
def fetch_transformers():
    """Fetch all transformers."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/transformers?limit=100", timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


@st.cache_data(ttl=60)
def fetch_dga_history(transformer_id: int):
    """Fetch DGA history for a transformer."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/dga/{transformer_id}/history", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_health_history(transformer_id: int):
    """Fetch health index history."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/health/{transformer_id}/history", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_gas_forecast(transformer_id: int, gas_type: str = "tdcg", months: int = 6):
    """Fetch gas trend forecast."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/predictions/{transformer_id}/forecast",
            params={"gas_type": gas_type, "forecast_months": months},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_health_forecast(transformer_id: int):
    """Fetch health index trend analysis."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/health/{transformer_id}/trend", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def main():
    """Main trend monitor page."""
    st.title("📈 Trend Monitor")
    st.markdown("Visualize gas trends, health index trends, and forecasts")

    # Get available transformers
    transformers = fetch_transformers()

    if not transformers:
        st.warning("No transformers available. Please ensure API is running.")
        return

    # Transformer selector
    transformer_options = {t["id"]: t["name"] for t in transformers}

    st.sidebar.markdown("### Trend Settings")

    selected_transformer = st.selectbox(
        "Select Transformer",
        options=list(transformer_options.keys()),
        format_func=lambda x: transformer_options[x],
    )

    # Fetch data for selected transformer
    with st.spinner("Loading trend data..."):
        dga_history = fetch_dga_history(selected_transformer)
        health_history = fetch_health_history(selected_transformer)
        gas_forecast = fetch_gas_forecast(selected_transformer, "tdcg", 6)
        health_forecast = fetch_health_forecast(selected_transformer)

    # Date range selector
    st.sidebar.markdown("#### Date Range")
    date_range = st.sidebar.select_slider(
        "Select time range",
        options=["3 Months", "6 Months", "1 Year", "2 Years", "All"],
        value="1 Year",
    )

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Health Index Trends", "Gas Trends", "Forecasts"])

    with tab1:
        st.markdown("### Health Index Trends")

        if health_history and health_history.get("records"):
            records = health_history["records"]

            # Convert to DataFrame
            hist_df = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(r["calculation_date"]),
                        "value": r["health_index"],
                        "dga_score": r.get("dga_score"),
                        "oil_score": r.get("oil_quality_score"),
                        "electrical_score": r.get("electrical_score"),
                    }
                    for r in records
                    if r.get("health_index")
                ]
            )

            if not hist_df.empty:
                # Filter by date range
                if date_range != "All":
                    days_map = {
                        "3 Months": 90,
                        "6 Months": 180,
                        "1 Year": 365,
                        "2 Years": 730,
                    }
                    days = days_map.get(date_range, 365)
                    cutoff_date = datetime.now() - timedelta(days=days)
                    hist_df = hist_df[hist_df["date"] >= cutoff_date]

                # Current health index
                current_idx = hist_df.iloc[-1]["value"]

                col1, col2 = st.columns([1, 2])

                with col1:
                    fig_gauge = create_health_gauge(
                        current_idx, "Current Health Index", height=250
                    )
                    st.plotly_chart(
                        fig_gauge,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                    # Trend info
                    if health_forecast:
                        st.metric(
                            "Trend",
                            health_forecast.get("trend_direction", "Unknown").title(),
                            delta_color="normal",
                        )
                        st.caption(
                            f"Monthly rate: {health_forecast.get('monthly_rate', 0):.2f}"
                        )

                with col2:
                    # Health trend chart
                    fig = create_gas_trend_chart(hist_df, "Health Index", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                # Component scores over time
                st.markdown("#### Component Scores Over Time")

                comp_df = hist_df[
                    ["date", "dga_score", "oil_score", "electrical_score"]
                ].copy()
                comp_df = comp_df.rename(
                    columns={
                        "dga_score": "DGA Score",
                        "oil_score": "Oil Quality",
                        "electrical_score": "Electrical",
                    }
                )

                if not comp_df.empty:
                    fig_comp = create_multi_gas_chart(
                        comp_df, ["DGA Score", "Oil Quality", "Electrical"], height=400
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No health index history available")
        else:
            st.info("No health index history available")

        # Forecast predictions
        if health_forecast:
            st.markdown("#### Health Index Forecast")

            pred_6m = health_forecast.get("predicted_index_6_months")
            pred_12m = health_forecast.get("predicted_index_12_months")

            col1, col2 = st.columns(2)

            with col1:
                if pred_6m is not None:
                    st.metric("Predicted (6 months)", f"{pred_6m:.1f}")
            with col2:
                if pred_12m is not None:
                    st.metric("Predicted (12 months)", f"{pred_12m:.1f}")

            if health_forecast.get("confidence"):
                st.caption(
                    f"Forecast confidence: {health_forecast['confidence'] * 100:.1f}%"
                )

    with tab2:
        st.markdown("### Gas Concentration Trends")

        if dga_history and dga_history.get("records"):
            records = dga_history["records"]

            # Convert to DataFrame
            gas_df = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(r["sample_date"]),
                        "h2": r.get("h2", 0),
                        "ch4": r.get("ch4", 0),
                        "c2h2": r.get("c2h2", 0),
                        "c2h4": r.get("c2h4", 0),
                        "c2h6": r.get("c2h6", 0),
                        "co": r.get("co", 0),
                        "co2": r.get("co2", 0),
                        "tdcg": r.get("tdcg", 0),
                    }
                    for r in records
                ]
            )

            if not gas_df.empty:
                # Filter by date range
                if date_range != "All":
                    days_map = {
                        "3 Months": 90,
                        "6 Months": 180,
                        "1 Year": 365,
                        "2 Years": 730,
                    }
                    days = days_map.get(date_range, 365)
                    cutoff_date = datetime.now() - timedelta(days=days)
                    gas_df = gas_df[gas_df["date"] >= cutoff_date]

                # Gas selector
                st.sidebar.markdown("#### Gas Selection")
                selected_gases = st.sidebar.multiselect(
                    "Select gases to display",
                    options=["h2", "ch4", "c2h2", "c2h4", "c2h6", "co", "co2", "tdcg"],
                    default=["tdcg"],
                )

                if selected_gases:
                    fig = create_multi_gas_chart(gas_df, selected_gases, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                # Individual gas charts
                st.markdown("#### Individual Gas Trends")

                for gas in ["h2", "ch4", "c2h2", "c2h4", "c2h6", "co", "co2"]:
                    if gas in gas_df.columns:
                        gas_data = gas_df[["date", gas]].copy()
                        gas_data = gas_data.rename(columns={gas: "value"})

                        # Skip if all values are 0
                        if gas_data["value"].sum() > 0:
                            with st.expander(f"📊 {gas.upper()} Trend"):
                                fig_single = create_gas_trend_chart(
                                    gas_data, gas.upper(), height=300
                                )
                                st.plotly_chart(fig_single, use_container_width=True)
            else:
                st.info("No DGA history available")
        else:
            st.info("No DGA history available")

    with tab3:
        st.markdown("### Forecast Analysis")

        # TDCG Forecast
        st.markdown("#### Total Gas (TDCG) Forecast")

        if gas_forecast and not gas_forecast.get("error"):
            trend = gas_forecast.get("trend", "Unknown")
            slope = gas_forecast.get("trend_slope", 0)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Trend", trend.title())
            with col2:
                st.metric("Trend Slope", f"{slope:.2f} ppm/month")
            with col3:
                st.metric("Data Points", gas_forecast.get("data_points", 0))

            # Create forecast chart
            historical = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(r["sample_date"]),
                        "value": r.get("tdcg", 0),
                    }
                    for r in dga_history.get("records", [])
                    if r.get("tdcg")
                ]
            )

            forecast_dates = gas_forecast.get("forecast_dates", [])
            forecast_values = gas_forecast.get("forecast_values", [])

            forecast = pd.DataFrame(
                {
                    "date": [pd.to_datetime(d) for d in forecast_dates],
                    "value": forecast_values,
                }
            )

            if not historical.empty:
                fig_forecast = create_forecast_chart(
                    historical, forecast, "TDCG", height=400
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

            # Forecast values table
            if forecast_values:
                st.markdown("#### Forecast Values")

                forecast_table = pd.DataFrame(
                    {
                        "Date": [d[:10] for d in forecast_dates],
                        "Predicted TDCG (ppm)": [f"{v:.1f}" for v in forecast_values],
                    }
                )
                st.table(forecast_table)
        else:
            st.info("Not enough data for forecasting (minimum 3 DGA records required)")

        # Anomaly detection
        st.markdown("---")
        st.markdown("#### Anomaly Detection")

        if gas_df is not None and not gas_df.empty:
            # Simple anomaly detection - values above threshold
            thresholds = {
                "h2": 100,
                "ch4": 120,
                "c2h2": 50,
                "c2h4": 50,
                "co": 350,
                "co2": 2500,
            }

            anomalies_found = []

            for gas, threshold in thresholds.items():
                if gas in gas_df.columns:
                    above_threshold = gas_df[gas_df[gas] > threshold]
                    if not above_threshold.empty:
                        for _, row in above_threshold.iterrows():
                            anomalies_found.append(
                                {
                                    "Date": row["date"].strftime("%Y-%m-%d"),
                                    "Gas": gas.upper(),
                                    "Value": row[gas],
                                    "Threshold": threshold,
                                }
                            )

            if anomalies_found:
                st.warning(f"Found {len(anomalies_found)} anomaly/anomalies:")

                anomaly_df = pd.DataFrame(anomalies_found)
                st.table(anomaly_df)
            else:
                st.success(
                    "No anomalies detected - all gas values within normal ranges"
                )
        else:
            st.info("No data available for anomaly detection")


if __name__ == "__main__":
    main()
