"""
Interactive Duval Triangle Visualization
DGA fault diagnosis using the Duval Triangle method
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# Duval Triangle fault zone definitions
# Format: (name, color, vertices as [ch4, c2h4, c2h2] percentages)
DUVAL_ZONES = {
    "PD": {  # Partial Discharge
        "name": "PD - Partial Discharge",
        "color": "#3498db",
        "vertices": [(0, 0, 100), (0, 20, 80), (20, 0, 80), (0, 0, 100)],
    },
    "D1": {  # Low-energy discharge
        "name": "D1 - Low Energy Discharge",
        "color": "#e74c3c",
        "vertices": [(20, 0, 80), (20, 50, 30), (50, 20, 30), (20, 0, 80)],
    },
    "D2": {  # High-energy discharge
        "name": "D2 - High Energy Discharge",
        "color": "#c0392b",
        "vertices": [(20, 50, 30), (50, 80, 0), (80, 20, 0), (20, 50, 30)],
    },
    "T1": {  # Low-temperature thermal
        "name": "T1 - Thermal Fault <300°C",
        "color": "#f39c12",
        "vertices": [(0, 80, 20), (20, 50, 30), (20, 100, 0), (0, 80, 20)],
    },
    "T2": {  # Medium-temperature thermal
        "name": "T2 - Thermal Fault 300-700°C",
        "color": "#e67e22",
        "vertices": [(20, 50, 30), (20, 100, 0), (50, 100, 0), (20, 50, 30)],
    },
    "T3": {  # High-temperature thermal
        "name": "T3 - Thermal Fault >700°C",
        "color": "#d35400",
        "vertices": [(50, 100, 0), (80, 100, 0), (50, 50, 0), (50, 100, 0)],
    },
    "DT": {  # Mixed discharge/thermal
        "name": "DT - Mixed Thermal/Discharge",
        "color": "#9b59b6",
        "vertices": [
            (20, 0, 80),
            (20, 20, 60),
            (50, 20, 30),
            (20, 50, 30),
            (20, 0, 80),
        ],
    },
}


def calculate_duval_percentages(
    ch4: float, c2h4: float, c2h2: float
) -> Tuple[float, float, float]:
    """
    Calculate percentages for Duval Triangle from gas concentrations.

    Args:
        ch4: Methane concentration (ppm)
        c2h4: Ethylene concentration (ppm)
        c2h2: Acetylene concentration (ppm)

    Returns:
        Tuple of (CH4%, C2H4%, C2H2%)
    """
    total = ch4 + c2h4 + c2h2

    if total == 0:
        return 0, 0, 0

    ch4_pct = (ch4 / total) * 100
    c2h4_pct = (c2h4 / total) * 100
    c2h2_pct = (c2h2 / total) * 100

    return ch4_pct, c2h4_pct, c2h2_pct


def get_fault_zone(ch4_pct: float, c2h4_pct: float, c2h2_pct: float) -> str:
    """
    Determine the fault zone based on Duval Triangle method.

    Args:
        ch4_pct: Methane percentage
        c2h4_pct: Ethylene percentage
        c2h2_pct: Acetylene percentage

    Returns:
        Fault zone name
    """
    # Simplified zone determination based on Duval Triangle

    # PD zone (Partial Discharge)
    if c2h2_pct >= 80:
        return "PD"

    # D1 zone (Low-energy discharge)
    if c2h2_pct >= 30 and c2h4_pct <= 50:
        if ch4_pct >= 20:
            return "D1"

    # D2 zone (High-energy discharge)
    if c2h4_pct >= 50 and c2h2_pct >= 20:
        if ch4_pct <= 50:
            return "D2"

    # T1 zone (<300°C)
    if c2h4_pct >= 50 and c2h4_pct <= 80:
        if c2h2_pct <= 20 and ch4_pct <= 50:
            return "T1"

    # T2 zone (300-700°C)
    if c2h4_pct >= 50 and ch4_pct >= 20:
        if c2h2_pct <= 30:
            return "T2"

    # T3 zone (>700°C)
    if c2h4_pct >= 80 and c2h2_pct <= 20:
        return "T3"

    # DT zone (Mixed)
    if c2h2_pct >= 20 and c2h2_pct < 80:
        if ch4_pct <= 50 and c2h4_pct <= 50:
            return "DT"

    # Default - check for thermal
    if c2h4_pct > ch4_pct:
        if c2h2_pct < 20:
            return "T2" if ch4_pct >= 20 else "T1"

    return "Unknown"


def create_duval_triangle(
    ch4_pct: float = 0,
    c2h4_pct: float = 0,
    c2h2_pct: float = 0,
    show_zones: bool = True,
    height: int = 600,
) -> go.Figure:
    """
    Create interactive Duval Triangle plot with fault zones.

    Args:
        ch4_pct: Methane percentage
        c2h4_pct: Ethylene percentage
        c2h2_pct: Acetylene percentage
        show_zones: Whether to show fault zone labels
        height: Figure height in pixels

    Returns:
        Plotly Figure with triangle zones and current point
    """
    fig = go.Figure()

    # Triangle vertices (equilateral triangle)
    # Top: C2H2, Bottom-left: CH4, Bottom-right: C2H4
    x_left = 0
    y_bottom = 0
    x_right = 100
    y_top = 86.6  # Height of equilateral triangle

    # Draw fault zones if enabled
    if show_zones:
        # PD Zone (top area)
        fig.add_trace(
            go.Scatter(
                x=[0, 0, 20, 20, 0],
                y=[0, y_top * 0.8, y_top * 0.8, 0, 0],
                fill="toself",
                fillcolor="rgba(52, 152, 219, 0.3)",
                line=dict(color="#3498db", width=2),
                name="PD",
                hoverinfo="name",
            )
        )

        # D1 Zone (left side)
        fig.add_trace(
            go.Scatter(
                x=[0, 20, 50, 20, 0],
                y=[0, y_top * 0.8, y_top * 0.3, 0, 0],
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.3)",
                line=dict(color="#e74c3c", width=2),
                name="D1",
                hoverinfo="name",
            )
        )

        # D2 Zone (center-right)
        fig.add_trace(
            go.Scatter(
                x=[20, 50, 80, 20],
                y=[y_top * 0.8, y_top * 0.3, y_top * 0.8, y_top * 0.8],
                fill="toself",
                fillcolor="rgba(192, 57, 43, 0.3)",
                line=dict(color="#c0392b", width=2),
                name="D2",
                hoverinfo="name",
            )
        )

        # T1 Zone (bottom-left)
        fig.add_trace(
            go.Scatter(
                x=[0, 20, 20, 0],
                y=[0, 0, y_top * 0.3, 0],
                fill="toself",
                fillcolor="rgba(243, 156, 18, 0.3)",
                line=dict(color="#f39c12", width=2),
                name="T1",
                hoverinfo="name",
            )
        )

        # T2 Zone (bottom-center)
        fig.add_trace(
            go.Scatter(
                x=[20, 50, 50, 20],
                y=[0, 0, y_top * 0.3, y_top * 0.3],
                fill="toself",
                fillcolor="rgba(230, 126, 34, 0.3)",
                line=dict(color="#e67e22", width=2),
                name="T2",
                hoverinfo="name",
            )
        )

        # T3 Zone (bottom-right)
        fig.add_trace(
            go.Scatter(
                x=[50, 80, 50],
                y=[0, y_top * 0.8, 0],
                fill="toself",
                fillcolor="rgba(211, 84, 0, 0.3)",
                line=dict(color="#d35400", width=2),
                name="T3",
                hoverinfo="name",
            )
        )

        # DT Zone (center)
        fig.add_trace(
            go.Scatter(
                x=[20, 50, 50, 20],
                y=[y_top * 0.3, y_top * 0.3, y_top * 0.8, y_top * 0.8],
                fill="toself",
                fillcolor="rgba(155, 89, 182, 0.3)",
                line=dict(color="#9b59b6", width=2),
                name="DT",
                hoverinfo="name",
            )
        )

    # Draw triangle outline
    fig.add_trace(
        go.Scatter(
            x=[0, 100, 50, 0],
            y=[0, 0, y_top, 0],
            mode="lines",
            line=dict(color="#2c3e50", width=3),
            name="Triangle",
            hoverinfo="skip",
        )
    )

    # Add axis labels
    fig.add_annotation(x=-5, y=0, text="CH4%", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=105, y=0, text="C2H4%", showarrow=False, font=dict(size=14))
    fig.add_annotation(
        x=50, y=y_top + 5, text="C2H2%", showarrow=False, font=dict(size=14)
    )

    # Add current operating point if values provided
    if ch4_pct > 0 or c2h4_pct > 0 or c2h2_pct > 0:
        # Calculate coordinates for the point
        # In Duval triangle: x = C2H4%, y relates to CH4 and C2H2
        point_x = c2h4_pct
        point_y = ch4_pct * (y_top / 100) + c2h2_pct * (y_top / 100) * 0.5

        fig.add_trace(
            go.Scatter(
                x=[point_x],
                y=[point_y],
                mode="markers+text",
                marker=dict(size=20, color="#e74c3c", symbol="diamond"),
                text=[
                    f"Current<br>CH4:{ch4_pct:.1f}%<br>C2H4:{c2h4_pct:.1f}%<br>C2H2:{c2h2_pct:.1f}%"
                ],
                textposition="top center",
                name="Current Point",
            )
        )

        # Determine and display fault zone
        fault_zone = get_fault_zone(ch4_pct, c2h4_pct, c2h2_pct)

        # Add fault zone annotation
        fig.add_annotation(
            x=50,
            y=y_top - 10,
            text=f"Fault Zone: <b>{fault_zone}</b>",
            showarrow=False,
            font=dict(size=16, color="#2c3e50"),
            bgcolor="white",
            bordercolor="#2c3e50",
            borderwidth=1,
            borderpad=4,
        )

    # Update layout
    fig.update_layout(
        height=height,
        xaxis=dict(
            range=[-10, 120],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="C2H4 %",
        ),
        yaxis=dict(
            range=[-10, 100],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def create_duval_legend() -> go.Figure:
    """
    Create a legend figure showing fault zone colors.

    Returns:
        Plotly Figure with legend
    """
    zones = [
        ("PD", "Partial Discharge", "#3498db"),
        ("D1", "Low Energy Discharge", "#e74c3c"),
        ("D2", "High Energy Discharge", "#c0392b"),
        ("T1", "Thermal Fault <300°C", "#f39c12"),
        ("T2", "Thermal Fault 300-700°C", "#e67e22"),
        ("T3", "Thermal Fault >700°C", "#d35400"),
        ("DT", "Mixed Discharge/Thermal", "#9b59b6"),
    ]

    fig = go.Figure()

    for code, name, color in zones:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=15, color=color),
                name=f"{code}: {name}",
            )
        )

    fig.update_layout(
        height=200,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=0.05),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
    )

    return fig
