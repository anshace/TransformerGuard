"""
Gas Concentration Charts Component
Time series visualization for DGA gas trends
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Standard gas colors for consistency
GAS_COLORS = {
    "h2": "#3498db",  # Hydrogen - Blue
    "ch4": "#2ecc71",  # Methane - Green
    "c2h2": "#e74c3c",  # Acetylene - Red
    "c2h4": "#9b59b6",  # Ethylene - Purple
    "c2h6": "#f39c12",  # Ethane - Orange
    "co": "#1abc9c",  # Carbon Monoxide - Teal
    "co2": "#34495e",  # Carbon Dioxide - Dark Gray
    "o2": "#95a5a6",  # Oxygen - Light Gray
    "n2": "#bdc3c7",  # Nitrogen - Silver
    "tdcg": "#e67e22",  # Total Dissolved Combustible Gas - Orange
}

# Gas display names
GAS_NAMES = {
    "h2": "Hydrogen (H2)",
    "ch4": "Methane (CH4)",
    "c2h2": "Acetylene (C2H2)",
    "c2h4": "Ethylene (C2H4)",
    "c2h6": "Ethane (C2H6)",
    "co": "Carbon Monoxide (CO)",
    "co2": "Carbon Dioxide (CO2)",
    "o2": "Oxygen (O2)",
    "n2": "Nitrogen (N2)",
    "tdcg": "Total Dissolved Combustible Gas (TDCG)",
}


def create_gas_trend_chart(
    df: pd.DataFrame,
    gas_name: str,
    threshold: Optional[float] = None,
    show_anomalies: bool = True,
    height: int = 400,
) -> go.Figure:
    """
    Create time series chart for gas concentration.

    Args:
        df: DataFrame with columns ['date', 'value', 'threshold' (optional)]
        gas_name: Name of gas for title
        threshold: Optional threshold line value
        show_anomalies: Whether to highlight anomalies
        height: Figure height in pixels

    Returns:
        Plotly Figure with trend line and threshold
    """
    if df.empty or len(df) == 0:
        fig = go.Figure()
        fig.update_layout(
            height=height,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            annotations=[
                dict(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date")

    # Get gas color
    color = GAS_COLORS.get(gas_name.lower(), "#3498db")
    display_name = GAS_NAMES.get(gas_name.lower(), gas_name)

    fig = go.Figure()

    # Add main trend line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines+markers",
            name=display_name,
            line=dict(color=color, width=2),
            marker=dict(size=6, symbol="circle"),
            hovertemplate=f"<b>{display_name}</b><br>"
            + "Date: %{x|%Y-%m-%d}<br>"
            + "Value: %{y:.1f} ppm<extra></extra>",
        )
    )

    # Add threshold line if provided
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#e74c3c",
            line_width=2,
            annotation_text=f"Threshold: {threshold}",
            annotation_position="top right",
        )

    # Highlight anomalies if enabled and threshold provided
    if show_anomalies and threshold is not None:
        anomalies = df[df["value"] > threshold]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["date"],
                    y=anomalies["value"],
                    mode="markers",
                    name="Above Threshold",
                    marker=dict(
                        size=12,
                        color="#e74c3c",
                        symbol="x",
                        line=dict(width=2, color="#c0392b"),
                    ),
                    hovertemplate="<b>ANOMALY</b><br>"
                    + "Date: %{x|%Y-%m-%d}<br>"
                    + "Value: %{y:.1f} ppm<extra></extra>",
                )
            )

    # Update layout
    fig.update_layout(
        height=height,
        xaxis=dict(
            title="Date", showgrid=True, gridcolor="rgba(0,0,0,0.1)", zeroline=False
        ),
        yaxis=dict(
            title="Concentration (ppm)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=50, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )

    return fig


def create_multi_gas_chart(
    df: pd.DataFrame, gas_list: List[str], height: int = 500
) -> go.Figure:
    """
    Create multi-gas time series chart.

    Args:
        df: DataFrame with gas columns and 'date' column
        gas_list: List of gas names to plot
        height: Figure height in pixels

    Returns:
        Plotly Figure with multiple gas trends
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=height,
            annotations=[
                dict(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
        return fig

    # Ensure date column is datetime
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    fig = go.Figure()

    for gas in gas_list:
        if gas in df.columns:
            color = GAS_COLORS.get(gas.lower(), "#3498db")
            name = GAS_NAMES.get(gas.lower(), gas)

            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[gas],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                )
            )

    fig.update_layout(
        height=height,
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(
            title="Concentration (ppm)", showgrid=True, gridcolor="rgba(0,0,0,0.1)"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )

    return fig


def create_gas_bar_chart(
    df: pd.DataFrame,
    gas_column: str = "value",
    title: str = "Gas Distribution",
    height: int = 400,
) -> go.Figure:
    """
    Create bar chart for gas distribution.

    Args:
        df: DataFrame with gas data
        gas_column: Column name for gas values
        title: Chart title
        height: Figure height in pixels

    Returns:
        Plotly Figure with bar chart
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=height,
            annotations=[
                dict(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
        return fig

    # Get gas values and names
    gas_values = []
    gas_names = []
    colors = []

    for gas in ["h2", "ch4", "c2h2", "c2h4", "c2h6", "co", "co2"]:
        if gas in df.columns:
            value = df[gas].iloc[-1] if len(df) > 0 else 0
            if value is not None:
                gas_values.append(value)
                gas_names.append(GAS_NAMES.get(gas, gas))
                colors.append(GAS_COLORS.get(gas, "#3498db"))

    fig = go.Figure(
        data=[
            go.Bar(
                x=gas_names,
                y=gas_values,
                marker_color=colors,
                text=[f"{v:.1f}" for v in gas_values],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        height=height,
        title=title,
        xaxis=dict(title="Gas Type"),
        yaxis=dict(title="Concentration (ppm)"),
        margin=dict(l=60, r=30, t=50, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def create_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    gas_name: str,
    height: int = 400,
) -> go.Figure:
    """
    Create chart with historical data and forecast.

    Args:
        historical_df: Historical gas data
        forecast_df: Forecasted gas data
        gas_name: Name of gas being forecasted
        height: Figure height in pixels

    Returns:
        Plotly Figure with historical and forecast data
    """
    color = GAS_COLORS.get(gas_name.lower(), "#3498db")
    display_name = GAS_NAMES.get(gas_name.lower(), gas_name)

    fig = go.Figure()

    # Historical data
    if not historical_df.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["value"],
                mode="lines+markers",
                name="Historical",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )

    # Forecast data
    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["value"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
            )
        )

        # Connect historical to forecast
        if not historical_df.empty and not forecast_df.empty:
            last_historical = historical_df.iloc[-1]
            first_forecast = forecast_df.iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=[last_historical["date"], first_forecast["date"]],
                    y=[last_historical["value"], first_forecast["value"]],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        height=height,
        title=f"{display_name} - Historical & Forecast",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Concentration (ppm)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=50, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )

    return fig
