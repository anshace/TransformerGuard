"""
Dashboard Components
Reusable visualization components for the TransformerGuard dashboard
"""

from .duval_plot import (
    DUVAL_ZONES,
    calculate_duval_percentages,
    create_duval_legend,
    create_duval_triangle,
    get_fault_zone,
)
from .gas_chart import (
    GAS_COLORS,
    GAS_NAMES,
    create_forecast_chart,
    create_gas_bar_chart,
    create_gas_trend_chart,
    create_multi_gas_chart,
)
from .health_gauge import (
    HEALTH_COLORS,
    create_health_gauge,
    create_mini_gauge,
    get_health_category,
    get_health_color,
    get_health_status_icon,
)

__all__ = [
    # Health gauge
    "create_health_gauge",
    "create_mini_gauge",
    "get_health_color",
    "get_health_category",
    "get_health_status_icon",
    "HEALTH_COLORS",
    # Duval triangle
    "create_duval_triangle",
    "calculate_duval_percentages",
    "get_fault_zone",
    "create_duval_legend",
    "DUVAL_ZONES",
    # Gas charts
    "create_gas_trend_chart",
    "create_multi_gas_chart",
    "create_gas_bar_chart",
    "create_forecast_chart",
    "GAS_COLORS",
    "GAS_NAMES",
]
