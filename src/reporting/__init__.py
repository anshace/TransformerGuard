"""
Report Generator Module
Template-based explanations, alerts, recommendations, and PDF reports
"""

from .action_recommender import ActionRecommender, Recommendation
from .alert_generator import Alert, AlertGenerator, AlertPriority
from .pdf_report import PDFReportGenerator, ReportConfig
from .template_engine import TemplateEngine, TemplateResult

__all__ = [
    "TemplateEngine",
    "TemplateResult",
    "AlertGenerator",
    "Alert",
    "AlertPriority",
    "ActionRecommender",
    "Recommendation",
    "PDFReportGenerator",
    "ReportConfig",
]
