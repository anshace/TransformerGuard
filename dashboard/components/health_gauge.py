"""
Health Index Gauge Component
Reusable gauge widget for displaying health index (0-100)
"""

from typing import Optional

import plotly.graph_objects as go

# Health category colors
HEALTH_COLORS = {
    "EXCELLENT": "#2ecc71",  # Green (85-100)
    "GOOD": "#27ae60",  # Light Green (70-84)
    "FAIR": "#f39c12",  # Yellow/Orange (50-69)
    "POOR": "#e67e22",  # Orange (25-49)
    "CRITICAL": "#e74c3c",  # Red (0-24)
}

# Reverse mapping for score lookup
SCORE_ZONES = [
    (85, 100, "EXCELLENT", "#2ecc71"),
    (70, 84, "GOOD", "#27ae60"),
    (50, 69, "FAIR", "#f39c12"),
    (25, 49, "POOR", "#e67e22"),
    (0, 24, "CRITICAL", "#e74c3c"),
]


def get_health_category(score: float) -> tuple[str, str]:
    """
    Get health category and color for a given score.

    Args:
        score: Health index value (0-100)

    Returns:
        Tuple of (category, color)
    """
    for min_val, max_val, category, color in SCORE_ZONES:
        if min_val <= score <= max_val:
            return category, color
    return "UNKNOWN", "#95a5a6"


def create_health_gauge(
    score: float,
    title: str = "Health Index",
    show_needle: bool = True,
    show_scale: bool = True,
    height: int = 300,
) -> go.Figure:
    """
    Create a semi-circular gauge for health index display.

    Args:
        score: Health index value (0-100)
        title: Gauge title
        show_needle: Whether to show the needle pointer
        show_scale: Whether to show the scale labels
        height: Figure height in pixels

    Returns:
        Plotly Figure object
    """
    # Clamp score to valid range
    score = max(0, min(100, score))

    # Get category and color
    category, color = get_health_category(score)

    # Create the gauge
    fig = go.Figure()

    # Add the main gauge with colored zones
    fig.add_trace(
        go.Indicator(
            mode="gauge+number" if show_needle else "number",
            value=score,
            number={"suffix": f" ({category})", "font": {"size": 24, "color": color}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#2c3e50",
                    "tickmode": "array",
                    "tickvals": [0, 25, 50, 70, 85, 100],
                    "ticktext": ["0", "25", "50", "70", "85", "100"],
                },
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#34495e",
                "steps": [
                    {"range": [0, 24], "color": "#fadbd8"},
                    {"range": [24, 49], "color": "#fdebd0"},
                    {"range": [49, 69], "color": "#f9e79f"},
                    {"range": [69, 84], "color": "#d5f5e3"},
                    {"range": [84, 100], "color": "#d4efdf"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 2},
                    "thickness": 0.15,
                    "value": score,
                },
            },
            title={"text": title, "font": {"size": 18}},
        )
    )

    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={"color": "#2c3e50"},
    )

    return fig


def create_mini_gauge(score: float, width: int = 150, height: int = 100) -> go.Figure:
    """
    Create a smaller gauge for compact display.

    Args:
        score: Health index value (0-100)
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        Plotly Figure object
    """
    score = max(0, min(100, score))
    category, color = get_health_category(score)

    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 16, "color": color}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0.5,
                    "tickcolor": "#2c3e50",
                    "tickvals": [0, 50, 100],
                    "ticktext": ["0", "50", "100"],
                },
                "bar": {"color": color, "thickness": 0.4},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#bdc3c7",
                "steps": [
                    {"range": [0, 24], "color": "#fadbd8"},
                    {"range": [24, 49], "color": "#fdebd0"},
                    {"range": [49, 69], "color": "#f9e79f"},
                    {"range": [69, 84], "color": "#d5f5e3"},
                    {"range": [84, 100], "color": "#d4efdf"},
                ],
            },
        )
    )

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        font={"color": "#2c3e50"},
    )

    return fig


def get_health_color(score: float) -> str:
    """
    Get the color for a health score.

    Args:
        score: Health index value (0-100)

    Returns:
        Hex color code
    """
    _, color = get_health_category(score)
    return color


def get_health_status_icon(score: float) -> str:
    """
    Get a status icon for the health score.

    Args:
        score: Health index value (0-100)

    Returns:
        Emoji icon
    """
    category, _ = get_health_category(score)

    status_icons = {
        "EXCELLENT": "✅",
        "GOOD": "👍",
        "FAIR": "⚠️",
        "POOR": "🔶",
        "CRITICAL": "🔴",
    }

    return status_icons.get(category, "❓")
