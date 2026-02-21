"""
Alert Center Page
Alert management dashboard with active alerts, acknowledgment, and history
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

from components.health_gauge import get_health_color

# API Configuration
API_BASE_URL = "http://localhost:8000"


@st.cache_data(ttl=30)
def fetch_alerts(
    priority: str = None,
    category: str = None,
    acknowledged: bool = None,
    limit: int = 100,
):
    """Fetch alerts from API."""
    params = {"limit": limit}
    if priority:
        params["priority"] = priority
    if category:
        params["category"] = category
    if acknowledged is not None:
        params["acknowledged"] = acknowledged

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/alerts", params=params, timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=30)
def fetch_active_alerts(limit: int = 50):
    """Fetch active (unacknowledged) alerts."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/alerts/active", params={"limit": limit}, timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_alert_summary():
    """Fetch alert summary statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/alerts/summary", timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_transformer(transformer_id: int):
    """Fetch transformer details."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/transformers/{transformer_id}", timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def acknowledge_alert(alert_id: int, acknowledged_by: str):
    """Acknowledge an alert."""
    try:
        response = requests.put(
            f"{API_BASE_URL}/api/v1/alerts/{alert_id}/acknowledge",
            json={"acknowledged_by": acknowledged_by},
            timeout=30,
        )
        return response.status_code == 200
    except:
        return False


def main():
    """Main alert center page."""
    st.title("🔔 Alert Center")
    st.markdown("Monitor and manage transformer alerts")

    # Fetch alert data
    with st.spinner("Loading alerts..."):
        active_alerts = fetch_active_alerts(limit=100)
        alert_summary = fetch_alert_summary()
        all_alerts = fetch_alerts(acknowledged=False, limit=100)
        historical_alerts = fetch_alerts(acknowledged=True, limit=50)

    # Sidebar filters
    st.sidebar.markdown("### Alert Filters")

    priority_filter = st.sidebar.selectbox(
        "Priority", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    )

    category_filter = st.sidebar.selectbox(
        "Category", ["All", "DGA", "THERMAL", "HEALTH", "LOADING", "ANOMALY"]
    )

    # Summary statistics
    st.markdown("### Alert Summary")

    if alert_summary:
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total Alerts", alert_summary.get("total_alerts", 0))
        with col2:
            st.metric("Active", alert_summary.get("active_alerts", 0))
        with col3:
            st.metric(
                "Critical",
                alert_summary.get("critical_count", 0),
                delta_color="inverse",
            )
        with col4:
            st.metric("High", alert_summary.get("high_count", 0), delta_color="off")
        with col5:
            st.metric("Medium", alert_summary.get("medium_count", 0), delta_color="off")
        with col6:
            st.metric("Low", alert_summary.get("low_count", 0), delta_color="normal")

        # Category breakdown
        if alert_summary.get("by_category"):
            st.markdown("#### Alerts by Category")

            category_data = alert_summary["by_category"]
            cat_df = pd.DataFrame(
                [{"Category": k.title(), "Count": v} for k, v in category_data.items()]
            )

            if not cat_df.empty:
                import plotly.express as px

                fig = px.bar(
                    cat_df, x="Category", y="Count", color="Category", text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, height=250, margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Active Alerts", "Alert History", "All Alerts"])

    with tab1:
        st.markdown("### Active Alerts")

        # Acknowledge input
        col_ack, col_btn = st.columns([2, 1])
        with col_ack:
            acknowledged_by = st.text_input("Your Name (for acknowledgment)", "")
        with col_btn:
            acknowledge_btn = st.button("Acknowledge Selected", type="primary")

        if active_alerts and active_alerts.get("alerts"):
            alerts = active_alerts["alerts"]

            # Priority colors
            priority_colors = {
                "CRITICAL": "#e74c3c",
                "HIGH": "#e67e22",
                "MEDIUM": "#f39c12",
                "LOW": "#3498db",
                "INFO": "#95a5a6",
            }

            # Filter alerts
            if priority_filter != "All":
                alerts = [a for a in alerts if a.get("priority") == priority_filter]
            if category_filter != "All":
                alerts = [a for a in alerts if a.get("category") == category_filter]

            # Display alerts
            if alerts:
                # Create selection for acknowledgment
                alert_options = {}

                for alert in alerts:
                    priority = alert.get("priority", "UNKNOWN")
                    color = priority_colors.get(priority, "#95a5a6")

                    # Get transformer name
                    transformer_name = f"Transformer #{alert.get('transformer_id')}"
                    transformer = fetch_transformer(alert.get("transformer_id"))
                    if transformer:
                        transformer_name = transformer.get("name", transformer_name)

                    # Format alert card
                    title = (
                        f"🔴 {alert.get('title')}"
                        if priority == "CRITICAL"
                        else f"🟠 {alert.get('title')}"
                    )

                    with st.expander(f"{title} - {priority}", expanded=False):
                        col_info1, col_info2 = st.columns(2)

                        with col_info1:
                            st.write(f"**Priority:** {priority}")
                            st.write(f"**Category:** {alert.get('category', 'N/A')}")
                        with col_info2:
                            st.write(f"**Transformer:** {transformer_name}")
                            st.write(f"**Created:** {alert.get('created_at', '')[:19]}")

                        if alert.get("message"):
                            st.write(f"**Message:** {alert.get('message')}")

                        # Checkbox for acknowledgment
                        alert_options[alert["id"]] = st.checkbox(
                            "Select to acknowledge", key=f"alert_{alert['id']}"
                        )

                # Handle acknowledgment
                if acknowledge_btn and acknowledged_by:
                    acknowledged_ids = [
                        aid for aid, selected in alert_options.items() if selected
                    ]

                    if acknowledged_ids:
                        success_count = 0
                        for alert_id in acknowledged_ids:
                            if acknowledge_alert(alert_id, acknowledged_by):
                                success_count += 1

                        if success_count > 0:
                            st.success(
                                f"Successfully acknowledged {success_count} alert(s)"
                            )
                            st.rerun()
                        else:
                            st.error("Failed to acknowledge alerts")
                    else:
                        st.warning("No alerts selected")
                elif acknowledge_btn and not acknowledged_by:
                    st.warning("Please enter your name to acknowledge alerts")
            else:
                st.success("No active alerts match the selected filters!")
        else:
            st.success("No active alerts - all clear! ✅")

    with tab2:
        st.markdown("### Alert History")

        if historical_alerts and historical_alerts.get("alerts"):
            alerts = historical_alerts["alerts"]

            # Display acknowledged alerts
            for alert in alerts:
                priority = alert.get("priority", "UNKNOWN")

                with st.expander(
                    f"✅ {alert.get('title')} - {priority}", expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(
                            f"**Acknowledged by:** {alert.get('acknowledged_by', 'N/A')}"
                        )
                    with col2:
                        if alert.get("acknowledged_at"):
                            st.write(
                                f"**Acknowledged at:** {alert['acknowledged_at'][:19]}"
                            )

                    st.caption(f"Original message: {alert.get('message', 'N/A')}")
        else:
            st.info("No historical alerts")

    with tab3:
        st.markdown("### All Alerts")

        if all_alerts and all_alerts.get("alerts"):
            alerts = all_alerts["alerts"]

            # Filter
            if priority_filter != "All":
                alerts = [a for a in alerts if a.get("priority") == priority_filter]
            if category_filter != "All":
                alerts = [a for a in alerts if a.get("category") == category_filter]

            # Create DataFrame for table
            alert_data = []

            for alert in alerts:
                transformer_name = f"#{alert.get('transformer_id')}"
                transformer = fetch_transformer(alert.get("transformer_id"))
                if transformer:
                    transformer_name = transformer.get("name", transformer_name)

                alert_data.append(
                    {
                        "ID": alert.get("id"),
                        "Transformer": transformer_name,
                        "Priority": alert.get("priority"),
                        "Category": alert.get("category"),
                        "Title": alert.get("title"),
                        "Status": "Active"
                        if not alert.get("acknowledged")
                        else "Acknowledged",
                        "Created": alert.get("created_at", "")[:10],
                    }
                )

            if alert_data:
                df = pd.DataFrame(alert_data)

                # Apply color coding
                def highlight_priority(val):
                    color = priority_colors.get(val, "")
                    return f"background-color: {color}; color: white" if color else ""

                st.dataframe(
                    df.style.map(highlight_priority, subset=["Priority"]),
                    use_container_width=True,
                    height=400,
                )
        else:
            st.info("No alerts available")


if __name__ == "__main__":
    main()
