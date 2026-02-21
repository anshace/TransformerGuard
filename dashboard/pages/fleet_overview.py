"""
Fleet Overview Page
Display all transformers with health status and quick filters
"""

# Import dashboard utilities
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent
sys.path.insert(0, str(dashboard_path))

from components.health_gauge import (
    create_mini_gauge,
    get_health_color,
    get_health_status_icon,
)

# API Configuration
API_BASE_URL = "http://localhost:8000"


@st.cache_data(ttl=60)
def fetch_fleet_health():
    """Fetch fleet health overview from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/health/fleet-overview", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=120)
def fetch_transformers():
    """Fetch all transformers from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/transformers?limit=1000", timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


@st.cache_data(ttl=60)
def fetch_transformer_summaries():
    """Fetch all transformer summaries with health data."""
    try:
        transformers = fetch_transformers()
        summaries = []

        for t in transformers:
            try:
                resp = requests.get(
                    f"{API_BASE_URL}/api/v1/transformers/{t['id']}/summary", timeout=10
                )
                if resp.status_code == 200:
                    summaries.append(resp.json())
            except:
                summaries.append(
                    {
                        "id": t.get("id"),
                        "name": t.get("name"),
                        "substation": t.get("substation"),
                        "health_index": None,
                        "health_category": None,
                        "alert_count": 0,
                    }
                )

        return summaries
    except:
        return []


def main():
    """Main fleet overview page."""
    st.title("⚡ Fleet Overview")
    st.markdown(
        "Monitor all transformers across your fleet with health index indicators"
    )

    # Fetch data
    with st.spinner("Loading fleet data..."):
        summaries = fetch_transformer_summaries()

    if not summaries:
        st.warning(
            "No transformer data available. Please ensure the API is running and data exists."
        )
        return

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(summaries)

    # Sidebar filters
    st.sidebar.markdown("### Filters")

    # Substation filter
    substations = ["All"] + sorted([s for s in df["substation"].unique() if s])
    selected_substation = st.sidebar.selectbox("Substation", substations)

    # Health category filter
    categories = ["All", "EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL"]
    selected_category = st.sidebar.selectbox("Health Category", categories)

    # Apply filters
    filtered_df = df.copy()
    if selected_substation != "All":
        filtered_df = filtered_df[filtered_df["substation"] == selected_substation]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["health_category"] == selected_category]

    # Summary statistics
    st.markdown("### Fleet Summary")

    # Calculate statistics
    total = len(filtered_df)
    excellent = len(filtered_df[filtered_df["health_category"] == "EXCELLENT"])
    good = len(filtered_df[filtered_df["health_category"] == "GOOD"])
    fair = len(filtered_df[filtered_df["health_category"] == "FAIR"])
    poor = len(filtered_df[filtered_df["health_category"] == "POOR"])
    critical = len(filtered_df[filtered_df["health_category"] == "CRITICAL"])

    # Display metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Transformers", total)
    with col2:
        st.metric("Excellent", excellent, delta_color="normal")
    with col3:
        st.metric("Good", good, delta_color="normal")
    with col4:
        st.metric("Fair", fair, delta_color="off")
    with col5:
        st.metric("Poor", poor, delta_color="off")
    with col6:
        st.metric("Critical", critical, delta_color="inverse")

    # Health distribution chart
    st.markdown("### Health Distribution")

    if total > 0:
        # Create distribution data
        dist_data = pd.DataFrame(
            {
                "Category": ["Excellent", "Good", "Fair", "Poor", "Critical"],
                "Count": [excellent, good, fair, poor, critical],
            }
        )

        # Display as bar chart
        import plotly.express as px

        colors = {
            "Excellent": "#2ecc71",
            "Good": "#27ae60",
            "Fair": "#f39c12",
            "Poor": "#e67e22",
            "Critical": "#e74c3c",
        }

        fig = px.bar(
            dist_data,
            x="Category",
            y="Count",
            color="Category",
            color_discrete_map=colors,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False, height=300, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Transformer list
    st.markdown(f"### Transformers ({len(filtered_df)})")

    # Search box
    search_term = st.text_input(
        "Search transformers", placeholder="Enter transformer name..."
    )

    if search_term:
        filtered_df = filtered_df[
            filtered_df["name"].str.contains(search_term, case=False, na=False)
        ]

    # Display transformers in a grid
    if not filtered_df.empty:
        # Sort by health index (worst first for priority attention)
        filtered_df = filtered_df.sort_values(
            by=["health_index"], ascending=True, na_position="last"
        )

        # Create rows of 3 transformers each
        for i in range(0, len(filtered_df), 3):
            cols = st.columns(3)

            for j, idx in enumerate(range(i, min(i + 3, len(filtered_df)))):
                row = filtered_df.iloc[idx]

                with cols[j]:
                    # Create transformer card
                    with st.container():
                        # Header with name and status
                        health_idx = row.get("health_index")
                        category = row.get("health_category")
                        alerts = row.get("alert_count", 0)

                        # Status indicator
                        if category:
                            icon = get_health_status_icon(
                                health_idx if health_idx else 0
                            )
                            status_color = get_health_color(
                                health_idx if health_idx else 50
                            )
                        else:
                            icon = "❓"
                            status_color = "#95a5a6"

                        st.markdown(f"#### {icon} {row.get('name', 'Unknown')}")

                        # Substation info
                        if row.get("substation"):
                            st.caption(f"📍 {row.get('substation')}")

                        # Health index
                        if health_idx is not None:
                            # Mini gauge
                            fig = create_mini_gauge(health_idx, width=200, height=120)
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )
                        else:
                            st.warning("No health data")

                        # Additional info
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            if row.get("rated_mva"):
                                st.caption(f"⚡ {row.get('rated_mva')} MVA")
                        with col_info2:
                            if alerts > 0:
                                st.caption(f"🔔 {alerts} alerts")

                        # Link to detail page (using query params)
                        st.markdown(
                            f"[View Details](/Transformer%20Detail?transformer={row.get('id')})"
                        )

                        st.markdown("---")

    else:
        st.info("No transformers match the selected filters.")


if __name__ == "__main__":
    main()
