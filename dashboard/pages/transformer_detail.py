"""
Transformer Detail Page
Individual transformer deep-dive with health index, DGA, and predictions
"""

# Import dashboard components
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

dashboard_path = Path(__file__).parent.parent
sys.path.insert(0, str(dashboard_path))

from components.duval_plot import calculate_duval_percentages, create_duval_triangle
from components.gas_chart import create_gas_bar_chart, create_gas_trend_chart
from components.health_gauge import (
    create_health_gauge,
    get_health_category,
    get_health_status_icon,
)

# API Configuration
API_BASE_URL = "http://localhost:8000"


@st.cache_data(ttl=60)
def fetch_transformer(transformer_id: int):
    """Fetch transformer details."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/transformers/{transformer_id}", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_transformer_summary(transformer_id: int):
    """Fetch transformer summary with health info."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/transformers/{transformer_id}/summary", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_health_index(transformer_id: int):
    """Fetch current health index."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/health/{transformer_id}", timeout=30
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
def fetch_dga_history(transformer_id: int):
    """Fetch DGA history."""
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
def fetch_rul_estimate(transformer_id: int):
    """Fetch RUL estimate."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/predictions/{transformer_id}/rul", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_failure_probability(transformer_id: int):
    """Fetch failure probability."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/predictions/{transformer_id}/failure-probability",
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_latest_dga(transformer_id: int):
    """Fetch latest DGA record."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/dga/{transformer_id}", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_transformer_alerts(transformer_id: int):
    """Fetch transformer alerts."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/alerts/{transformer_id}?limit=10", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


def main():
    """Main transformer detail page."""
    st.title("🔌 Transformer Detail")

    # Get transformer ID from query params or sidebar selector
    # First try to get from query params
    query_params = st.query_params
    transformer_id = query_params.get("transformer", None)

    if not transformer_id:
        # Show transformer selector
        st.info("Select a transformer from the Fleet Overview page or enter an ID.")

        # Allow manual entry
        transformer_id_input = st.number_input(
            "Enter Transformer ID", min_value=1, step=1, value=1
        )

        if st.button("Load Transformer"):
            transformer_id = transformer_id_input
            st.query_params["transformer"] = str(transformer_id)
            st.rerun()
        else:
            return
    else:
        transformer_id = int(transformer_id)

    # Fetch transformer data
    with st.spinner("Loading transformer data..."):
        transformer = fetch_transformer(transformer_id)
        summary = fetch_transformer_summary(transformer_id)
        health = fetch_health_index(transformer_id)
        health_history = fetch_health_history(transformer_id)
        dga_history = fetch_dga_history(transformer_id)
        rul = fetch_rul_estimate(transformer_id)
        failure_prob = fetch_failure_probability(transformer_id)
        latest_dga = fetch_latest_dga(transformer_id)
        alerts = fetch_transformer_alerts(transformer_id)

    if not transformer:
        st.error(f"Could not load transformer with ID {transformer_id}")
        return

    # Header with transformer name
    st.markdown(f"## {transformer.get('name', 'Unknown Transformer')}")

    if summary and summary.get("health_index"):
        category, _ = get_health_category(summary["health_index"])
        icon = get_health_status_icon(summary["health_index"])
        st.markdown(f"### {icon} Status: {category}")

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Transformer Information Panel
        st.markdown("### 📋 Transformer Information")

        info_data = {
            "Serial Number": transformer.get("serial_number", "N/A"),
            "Manufacturer": transformer.get("manufacturer", "N/A"),
            "Installation Date": str(transformer.get("installation_date", "N/A")),
            "Rated MVA": f"{transformer.get('rated_mva', 'N/A')}"
            if transformer.get("rated_mva")
            else "N/A",
            "Rated Voltage": f"{transformer.get('rated_voltage_kv', 'N/A')} kV"
            if transformer.get("rated_voltage_kv")
            else "N/A",
            "Cooling Type": transformer.get("cooling_type", "N/A"),
            "Location": transformer.get("location", "N/A"),
            "Substation": transformer.get("substation", "N/A"),
        }

        for key, value in info_data.items():
            st.write(f"**{key}:** {value}")

    with col2:
        # Current Health Index Gauge
        st.markdown("### 💚 Health Index")

        if health:
            health_idx = health.get("health_index", 0)
            fig = create_health_gauge(health_idx, title="Current Health", height=250)
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

            # Show component scores
            st.markdown("#### Component Scores")

            scores = {
                "DGA Score": health.get("dga_score"),
                "Oil Quality": health.get("oil_quality_score"),
                "Electrical": health.get("electrical_score"),
                "Age Score": health.get("age_score"),
                "Loading": health.get("loading_score"),
            }

            for name, score in scores.items():
                if score is not None:
                    st.progress(score / 100, text=f"{name}: {score:.1f}")
        else:
            st.warning("No health index data available")

    # Row 2: DGA Analysis and Predictions
    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        # Latest DGA Results
        st.markdown("### 🧪 Latest DGA Analysis")

        if latest_dga:
            # Gas values
            gases = {
                "Hydrogen (H2)": latest_dga.get("h2"),
                "Methane (CH4)": latest_dga.get("ch4"),
                "Acetylene (C2H2)": latest_dga.get("c2h2"),
                "Ethylene (C2H4)": latest_dga.get("c2h4"),
                "Ethane (C2H6)": latest_dga.get("c2h6"),
                "Carbon Monoxide (CO)": latest_dga.get("co"),
                "Carbon Dioxide (CO2)": latest_dga.get("co2"),
            }

            st.write("**Gas Concentrations (ppm):**")
            for gas, value in gases.items():
                if value is not None:
                    st.write(f"  {gas}: {value:.1f}")

            # Fault diagnosis
            if latest_dga.get("fault_type"):
                st.write(f"**Fault Type:** {latest_dga.get('fault_type')}")
            if latest_dga.get("fault_confidence"):
                st.write(
                    f"**Confidence:** {latest_dga.get('fault_confidence') * 100:.1f}%"
                )

            # Sample date
            if latest_dga.get("sample_date"):
                st.caption(f"Sample Date: {latest_dga.get('sample_date')[:10]}")

            # Duval Triangle visualization
            if all(latest_dga.get(g) for g in ["ch4", "c2h4", "c2h2"]):
                ch4, c2h4, c2h2 = (
                    latest_dga["ch4"],
                    latest_dga["c2h4"],
                    latest_dga["c2h2"],
                )
                ch4_pct, c2h4_pct, c2h2_pct = calculate_duval_percentages(
                    ch4, c2h4, c2h2
                )

                fig_duval = create_duval_triangle(
                    ch4_pct, c2h4_pct, c2h2_pct, height=400
                )
                st.plotly_chart(fig_duval, use_container_width=True)
        else:
            st.warning("No DGA data available")

    with col4:
        # Predictions
        st.markdown("### 📈 Predictions")

        # RUL Estimate
        if rul:
            st.markdown("#### Remaining Useful Life (RUL)")

            rul_years = rul.get("rul_years")
            rul_days = rul.get("rul_days")
            confidence = rul.get("confidence")

            if rul_years is not None:
                st.metric("RUL (Years)", f"{rul_years:.1f}")
            if rul_days is not None:
                st.metric("RUL (Days)", f"{rul_days:.0f}")
            if confidence is not None:
                st.caption(f"Confidence: {confidence * 100:.1f}%")

            if rul.get("end_of_life_date"):
                st.caption(f"End of Life: {rul['end_of_life_date'][:10]}")

        # Failure Probability
        if failure_prob:
            st.markdown("#### Failure Probability")

            prob_1yr = failure_prob.get("probability_1_year", 0) * 100
            prob_5yr = failure_prob.get("probability_5_years", 0) * 100
            risk_level = failure_prob.get("risk_level", "Unknown")

            st.metric("1-Year Probability", f"{prob_1yr:.2f}%")
            st.metric("5-Year Probability", f"{prob_5yr:.2f}%")
            st.write(f"**Risk Level:** {risk_level}")

    # Row 3: Recommendations and Alerts
    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        # Recommendations
        st.markdown("### 💡 Recommendations")

        if health and health.get("recommendations"):
            for rec in health["recommendations"]:
                st.write(f"• {rec}")
        else:
            st.info("No recommendations available")

    with col6:
        # Active Alerts
        st.markdown("### 🔔 Active Alerts")

        if alerts:
            for alert in alerts[:5]:  # Show first 5
                priority_color = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "🔵",
                }.get(alert.get("priority", ""), "⚪")

                st.write(f"{priority_color} **{alert.get('title')}**")
                if alert.get("message"):
                    st.caption(alert["message"][:100] + "...")
        else:
            st.success("No active alerts")

    # Historical Trends
    st.markdown("---")
    st.markdown("### 📊 Historical Trends")

    # Health Index Trend
    if health_history and health_history.get("records"):
        records = health_history["records"]
        if records:
            hist_df = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(r["calculation_date"]),
                        "value": r["health_index"],
                    }
                    for r in records
                    if r.get("health_index")
                ]
            )

            if not hist_df.empty:
                fig = create_gas_trend_chart(hist_df, "Health Index", height=350)
                st.plotly_chart(fig, use_container_width=True)

    # DGA Trend
    if dga_history and dga_history.get("records"):
        records = dga_history["records"]
        if records:
            dga_df = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(r["sample_date"]),
                        "h2": r.get("h2", 0),
                        "ch4": r.get("ch4", 0),
                        "c2h2": r.get("c2h2", 0),
                        "c2h4": r.get("c2h4", 0),
                        "co": r.get("co", 0),
                        "co2": r.get("co2", 0),
                    }
                    for r in records
                ]
            )

            if not dga_df.empty:
                gas_list = ["h2", "ch4", "c2h2", "c2h4", "co", "co2"]
                from components.gas_chart import create_multi_gas_chart

                fig = create_multi_gas_chart(dga_df, gas_list, height=400)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
