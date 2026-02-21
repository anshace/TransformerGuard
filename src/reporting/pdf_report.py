"""
PDF Report Generator
Generates professional PDF reports using reportlab
Includes executive summary, transformer details, DGA analysis, thermal analysis,
health index breakdown, predictions, and recommendations
"""

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Try to import reportlab, fallback to HTML generation if not available
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Flowable,
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Try to import matplotlib for charts
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class ReportConfig:
    """Configuration for PDF report generation"""

    include_charts: bool = True
    include_recommendations: bool = True
    include_historical_data: bool = True
    output_path: str = "transformer_report.pdf"
    format: str = "PDF"  # PDF or HTML
    page_size: str = "A4"
    report_title: str = "Transformer Condition Assessment Report"


class PDFReportGenerator:
    """
    PDF Report Generator
    Generates professional, utility-ready PDF reports
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the PDF report generator

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.styles = None
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        if not REPORTLAB_AVAILABLE:
            return

        # Title style
        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1a1a2e"),
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            )
        )

        # Section header style
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#16213e"),
                spaceAfter=10,
                spaceBefore=15,
                borderPadding=5,
                borderColor=colors.HexColor("#0f3460"),
                borderWidth=1,
                fontName="Helvetica-Bold",
            )
        )

        # Subsection header style
        self.styles.add(
            ParagraphStyle(
                name="SubsectionHeader",
                parent=self.styles["Heading3"],
                fontSize=12,
                textColor=colors.HexColor("#0f3460"),
                spaceAfter=8,
                spaceBefore=10,
                fontName="Helvetica-Bold",
            )
        )

        # Normal text style
        self.styles.add(
            ParagraphStyle(
                name="NormalText",
                parent=self.styles["Normal"],
                fontSize=10,
                spaceAfter=6,
                leading=14,
            )
        )

        # Alert style
        self.styles.add(
            ParagraphStyle(
                name="AlertText",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#c0392b"),
                spaceAfter=6,
                leading=14,
            )
        )

    def generate_report(
        self,
        transformer_data: Dict[str, Any],
        dga_results: Optional[Dict[str, Any]] = None,
        thermal_results: Optional[Dict[str, Any]] = None,
        health_results: Optional[Dict[str, Any]] = None,
        prediction_results: Optional[Dict[str, Any]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate comprehensive PDF report

        Args:
            transformer_data: Basic transformer information
            dga_results: DGA analysis results
            thermal_results: Thermal analysis results
            health_results: Health index results
            prediction_results: Prediction results (RUL, failure probability)
            alerts: Active alerts
            recommendations: Maintenance recommendations
            historical_data: Historical data for charts

        Returns:
            Path to generated report
        """
        if self.config.format.upper() == "HTML":
            return self._generate_html_report(
                transformer_data,
                dga_results,
                thermal_results,
                health_results,
                prediction_results,
                alerts,
                recommendations,
                historical_data,
            )

        if not REPORTLAB_AVAILABLE:
            # Fallback to HTML if reportlab not available
            self.config.format = "HTML"
            return self._generate_html_report(
                transformer_data,
                dga_results,
                thermal_results,
                health_results,
                prediction_results,
                alerts,
                recommendations,
                historical_data,
            )

        return self._generate_pdf_report(
            transformer_data,
            dga_results,
            thermal_results,
            health_results,
            prediction_results,
            alerts,
            recommendations,
            historical_data,
        )

    def _generate_pdf_report(
        self,
        transformer_data: Dict[str, Any],
        dga_results: Optional[Dict[str, Any]],
        thermal_results: Optional[Dict[str, Any]],
        health_results: Optional[Dict[str, Any]],
        prediction_results: Optional[Dict[str, Any]],
        alerts: Optional[List[Dict[str, Any]]],
        recommendations: Optional[List[Dict[str, Any]]],
        historical_data: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Generate PDF report using reportlab"""
        # Create document
        page_size = A4 if self.config.page_size == "A4" else letter
        doc = SimpleDocTemplate(
            self.config.output_path,
            pagesize=page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        # Build story (content)
        story = []

        # Title page
        story.extend(self._build_title_page(transformer_data))

        # Executive Summary
        story.extend(
            self._build_executive_summary(
                transformer_data, health_results, prediction_results, alerts
            )
        )

        # Transformer Details
        story.extend(self._build_transformer_details(transformer_data))

        # DGA Analysis
        if dga_results:
            story.extend(self._build_dga_section(dga_results))

        # Thermal Analysis
        if thermal_results:
            story.extend(self._build_thermal_section(thermal_results))

        # Health Index
        if health_results:
            story.extend(self._build_health_section(health_results))

        # Predictions
        if prediction_results:
            story.extend(self._build_predictions_section(prediction_results))

        # Charts (if enabled)
        if self.config.include_charts and historical_data and MATPLOTLIB_AVAILABLE:
            chart_path = self._generate_charts(
                historical_data, transformer_data.get("transformer_id", "report")
            )
            if chart_path:
                story.append(PageBreak())
                story.append(
                    Paragraph("Historical Trends", self.styles["SectionHeader"])
                )
                story.append(Image(chart_path, width=6 * inch, height=4 * inch))

        # Recommendations
        if self.config.include_recommendations and recommendations:
            story.extend(self._build_recommendations_section(recommendations))

        # Alerts
        if alerts:
            story.extend(self._build_alerts_section(alerts))

        # Build PDF
        doc.build(story)

        return self.config.output_path

    def _build_title_page(self, transformer_data: Dict[str, Any]) -> List:
        """Build title page"""
        story = []

        # Report title
        story.append(Spacer(1, 2 * inch))
        story.append(Paragraph(self.config.report_title, self.styles["ReportTitle"]))

        story.append(Spacer(1, 0.5 * inch))

        # Transformer name
        transformer_name = transformer_data.get(
            "name", transformer_data.get("transformer_id", "Unknown")
        )
        story.append(
            Paragraph(f"Transformer: {transformer_name}", self.styles["SectionHeader"])
        )

        story.append(Spacer(1, 0.3 * inch))

        # Report date
        story.append(
            Paragraph(
                f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
                self.styles["NormalText"],
            )
        )

        # Location if available
        if "location" in transformer_data:
            story.append(
                Paragraph(
                    f"Location: {transformer_data['location']}",
                    self.styles["NormalText"],
                )
            )

        # Rating if available
        if "rating" in transformer_data:
            story.append(
                Paragraph(
                    f"Rating: {transformer_data['rating']}", self.styles["NormalText"]
                )
            )

        story.append(Spacer(1, 2 * inch))

        # IEEE Standards note
        story.append(
            Paragraph(
                "This report is generated in accordance with IEEE C57.104 and IEEE C57.91 standards.",
                self.styles["NormalText"],
            )
        )

        story.append(PageBreak())

        return story

    def _build_executive_summary(
        self,
        transformer_data: Dict[str, Any],
        health_results: Optional[Dict[str, Any]],
        prediction_results: Optional[Dict[str, Any]],
        alerts: Optional[List[Dict[str, Any]]],
    ) -> List:
        """Build executive summary section"""
        story = []

        story.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))

        # Overall condition
        if health_results:
            health_score = health_results.get("health_score", 0)
            health_category = health_results.get("category", "Unknown")

            story.append(
                Paragraph(
                    f"<b>Overall Condition:</b> {health_category} (Health Index: {health_score:.1f}/100)",
                    self.styles["NormalText"],
                )
            )

            # Color code based on condition
            if health_score < 25:
                story.append(
                    Paragraph(
                        "⚠️ CRITICAL: Immediate action required",
                        self.styles["AlertText"],
                    )
                )
            elif health_score < 50:
                story.append(
                    Paragraph(
                        "⚠️ WARNING: Urgent maintenance recommended",
                        self.styles["AlertText"],
                    )
                )

        # RUL summary
        if prediction_results:
            rul = prediction_results.get("rul_days")
            if rul is not None:
                story.append(
                    Paragraph(
                        f"<b>Estimated Remaining Life:</b> {rul:.0f} days",
                        self.styles["NormalText"],
                    )
                )

            failure_prob = prediction_results.get("failure_probability")
            if failure_prob is not None:
                story.append(
                    Paragraph(
                        f"<b>12-Month Failure Probability:</b> {failure_prob * 100:.1f}%",
                        self.styles["NormalText"],
                    )
                )

        # Alert summary
        if alerts:
            critical_count = len([a for a in alerts if a.get("priority") == "CRITICAL"])
            high_count = len([a for a in alerts if a.get("priority") == "HIGH"])

            if critical_count > 0 or high_count > 0:
                story.append(
                    Paragraph(
                        f"<b>Active Alerts:</b> {critical_count} Critical, {high_count} High Priority",
                        self.styles["AlertText"],
                    )
                )

        story.append(Spacer(1, 0.2 * inch))

        return story

    def _build_transformer_details(self, transformer_data: Dict[str, Any]) -> List:
        """Build transformer details section"""
        story = []

        story.append(Paragraph("Transformer Information", self.styles["SectionHeader"]))

        # Create table with transformer details
        data = []

        for key, value in transformer_data.items():
            if key not in ["name"]:  # Skip name as it's in title
                label = key.replace("_", " ").title()
                data.append([label, str(value)])

        if data:
            table = Table(data, colWidths=[2 * inch, 4 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (0, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(table)

        story.append(Spacer(1, 0.2 * inch))

        return story

    def _build_dga_section(self, dga_results: Dict[str, Any]) -> List:
        """Build DGA analysis section"""
        story = []

        story.append(Paragraph("DGA Analysis", self.styles["SectionHeader"]))

        # Fault diagnosis
        if "fault_type" in dga_results:
            story.append(
                Paragraph(
                    f"<b>Primary Fault Type:</b> {dga_results['fault_type']}",
                    self.styles["NormalText"],
                )
            )

        if "fault_description" in dga_results:
            story.append(
                Paragraph(
                    f"Description: {dga_results['fault_description']}",
                    self.styles["NormalText"],
                )
            )

        # Gas concentrations table
        if "gas_concentrations" in dga_results:
            story.append(
                Paragraph("Gas Concentrations:", self.styles["SubsectionHeader"])
            )

            gases = dga_results["gas_concentrations"]
            thresholds = {
                "h2": 100,
                "ch4": 120,
                "c2h2": 2,
                "c2h4": 50,
                "c2h6": 65,
                "co": 350,
                "co2": 2500,
            }

            data = [["Gas", "Concentration (ppm)", "Threshold (ppm)", "Status"]]
            for gas, threshold in thresholds.items():
                if gas in gases:
                    conc = gases[gas]
                    status = "EXCEEDS" if conc > threshold else "Normal"
                    data.append([gas.upper(), f"{conc:.1f}", str(threshold), status])

            table = Table(data, colWidths=[1 * inch, 1.5 * inch, 1.5 * inch, 1 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ]
                )
            )
            story.append(table)

        # Risk level
        if "risk_level" in dga_results:
            story.append(
                Paragraph(
                    f"<b>Risk Level:</b> {dga_results['risk_level']}",
                    self.styles["NormalText"],
                )
            )

        story.append(Spacer(1, 0.2 * inch))

        return story

    def _build_thermal_section(self, thermal_results: Dict[str, Any]) -> List:
        """Build thermal analysis section"""
        story = []

        story.append(Paragraph("Thermal Analysis", self.styles["SectionHeader"]))

        # Hot-spot temperature
        if "hotspot_temp" in thermal_results:
            hotspot = thermal_results["hotspot_temp"]
            status = thermal_results.get("hotspot_status", "Normal")

            story.append(
                Paragraph(
                    f"<b>Hot-spot Temperature:</b> {hotspot:.1f}°C (Status: {status})",
                    self.styles["NormalText"],
                )
            )

            if hotspot > 140:
                story.append(
                    Paragraph(
                        "⚠️ EXCEEDS IEEE C57.91 emergency limit (140°C)",
                        self.styles["AlertText"],
                    )
                )
            elif hotspot > 110:
                story.append(
                    Paragraph(
                        "⚠️ EXCEEDS IEEE C57.91 normal limit (110°C)",
                        self.styles["AlertText"],
                    )
                )

        # Top-oil temperature
        if "topoil_temp" in thermal_results:
            story.append(
                Paragraph(
                    f"<b>Top-oil Temperature:</b> {thermal_results['topoil_temp']:.1f}°C",
                    self.styles["NormalText"],
                )
            )

        # Load percentage
        if "load_percent" in thermal_results:
            load = thermal_results["load_percent"]
            story.append(
                Paragraph(
                    f"<b>Load:</b> {load:.1f}% of rated capacity",
                    self.styles["NormalText"],
                )
            )

            if load > 100:
                story.append(
                    Paragraph("⚠️ LOAD EXCEEDS RATED CAPACITY", self.styles["AlertText"])
                )

        # Warnings
        if "warnings" in thermal_results and thermal_results["warnings"]:
            story.append(Paragraph("Warnings:", self.styles["SubsectionHeader"]))
            for warning in thermal_results["warnings"]:
                story.append(Paragraph(f"• {warning}", self.styles["NormalText"]))

        story.append(Spacer(1, 0.2 * inch))

        return story

    def _build_health_section(self, health_results: Dict[str, Any]) -> List:
        """Build health index section"""
        story = []

        story.append(Paragraph("Health Index Assessment", self.styles["SectionHeader"]))

        # Health score
        if "health_score" in health_results:
            score = health_results["health_score"]
            category = health_results.get("category", "Unknown")

            story.append(
                Paragraph(
                    f"<b>Health Index:</b> {score:.1f}/100 ({category})",
                    self.styles["NormalText"],
                )
            )

        # Condition description
        if "condition" in health_results:
            story.append(
                Paragraph(
                    f"Condition: {health_results['condition']}",
                    self.styles["NormalText"],
                )
            )

        # DGA status
        if "dga_status" in health_results:
            story.append(
                Paragraph(
                    f"<b>DGA Status:</b> {health_results['dga_status']}",
                    self.styles["NormalText"],
                )
            )

        story.append(Spacer(1, 0.2 * inch))

        return story

    def _build_predictions_section(self, prediction_results: Dict[str, Any]) -> List:
        """Build predictions section"""
        story = []

        story.append(Paragraph("Predictions & Forecasts", self.styles["SectionHeader"]))

        # RUL
        if "rul_days" in prediction_results:
            rul = prediction_results["rul_days"]
            status = prediction_results.get("rul_status", "Unknown")

            story.append(
                Paragraph(
                    f"<b>Estimated Remaining Useful Life:</b> {rul:.0f} days ({status})",
                    self.styles["NormalText"],
                )
            )

            if rul < 30:
                story.append(
                    Paragraph(
                        "⚠️ CRITICAL: Immediate action required",
                        self.styles["AlertText"],
                    )
                )

        # Failure probability
        if "failure_probability" in prediction_results:
            prob = prediction_results["failure_probability"]
            risk = prediction_results.get("risk_level", "Unknown")

            story.append(
                Paragraph(
                    f"<b>12-Month Failure Probability:</b> {prob * 100:.1f}% (Risk: {risk})",
                    self.styles["NormalText"],
                )
            )

            if prob > 0.5:
                story.append(Paragraph("⚠️ HIGH FAILURE RISK", self.styles["AlertText"]))

        # Trend
        if "trend" in prediction_results:
            story.append(
                Paragraph(
                    f"<b>Trend:</b> {prediction_results['trend']}",
                    self.styles["NormalText"],
                )
            )

        story.append(Spacer(1, 0.2 * inch))

        return story

    def _build_recommendations_section(
        self, recommendations: List[Dict[str, Any]]
    ) -> List:
        """Build recommendations section"""
        story = []

        story.append(
            Paragraph("Maintenance Recommendations", self.styles["SectionHeader"])
        )

        for i, rec in enumerate(recommendations, 1):
            priority = rec.get("priority", 3)
            priority_label = {
                1: "CRITICAL",
                2: "HIGH",
                3: "MEDIUM",
                4: "LOW",
                5: "INFO",
            }.get(priority, "UNKNOWN")

            story.append(
                Paragraph(
                    f"{i}. [{priority_label}] {rec.get('action', 'N/A')}",
                    self.styles["SubsectionHeader"],
                )
            )

            story.append(
                Paragraph(
                    f"<b>Timeframe:</b> {rec.get('timeframe', 'N/A')}",
                    self.styles["NormalText"],
                )
            )

            story.append(
                Paragraph(
                    f"<b>Rationale:</b> {rec.get('rationale', 'N/A')}",
                    self.styles["NormalText"],
                )
            )

            if rec.get("references"):
                story.append(
                    Paragraph(
                        f"<b>References:</b> {', '.join(rec['references'])}",
                        self.styles["NormalText"],
                    )
                )

            if rec.get("estimated_cost"):
                story.append(
                    Paragraph(
                        f"<b>Estimated Cost:</b> {rec['estimated_cost']}",
                        self.styles["NormalText"],
                    )
                )

            story.append(Spacer(1, 0.1 * inch))

        return story

    def _build_alerts_section(self, alerts: List[Dict[str, Any]]) -> List:
        """Build alerts section"""
        story = []

        story.append(Paragraph("Active Alerts", self.styles["SectionHeader"]))

        for alert in alerts:
            priority = alert.get("priority", "INFO")

            # Style based on priority
            if priority == "CRITICAL":
                style = self.styles["AlertText"]
            else:
                style = self.styles["NormalText"]

            story.append(
                Paragraph(f"<b>[{priority}]</b> {alert.get('title', 'N/A')}", style)
            )

            story.append(
                Paragraph(alert.get("message", "N/A"), self.styles["NormalText"])
            )

            if alert.get("actions"):
                story.append(
                    Paragraph("Recommended Actions:", self.styles["SubsectionHeader"])
                )
                for action in alert["actions"]:
                    story.append(Paragraph(f"• {action}", self.styles["NormalText"]))

            story.append(Spacer(1, 0.1 * inch))

        return story

    def _generate_charts(
        self, historical_data: List[Dict[str, Any]], transformer_id: str
    ) -> Optional[str]:
        """Generate charts using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(
                f"Transformer {transformer_id} - Historical Trends", fontsize=14
            )

            # Extract data
            dates = [d.get("date", d.get("timestamp", "")) for d in historical_data]

            # Health score trend
            health_scores = [d.get("health_score", 0) for d in historical_data]
            if health_scores:
                axes[0, 0].plot(
                    range(len(health_scores)), health_scores, "b-o", linewidth=2
                )
                axes[0, 0].set_title("Health Index Trend")
                axes[0, 0].set_xlabel("Sample")
                axes[0, 0].set_ylabel("Health Score")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_ylim(0, 100)

            # DGA trend (key gases)
            for gas in ["h2", "c2h2", "c2h4"]:
                gas_values = [d.get("gases", {}).get(gas, 0) for d in historical_data]
                if any(gas_values):
                    axes[0, 1].plot(
                        range(len(gas_values)),
                        gas_values,
                        "-o",
                        label=gas.upper(),
                        linewidth=2,
                    )
            axes[0, 1].set_title("Key Gas Trends")
            axes[0, 1].set_xlabel("Sample")
            axes[0, 1].set_ylabel("Concentration (ppm)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Temperature trend
            hotspot_temps = [d.get("hotspot_temp", 0) for d in historical_data]
            topoil_temps = [d.get("topoil_temp", 0) for d in historical_data]
            if any(hotspot_temps):
                axes[1, 0].plot(
                    range(len(hotspot_temps)),
                    hotspot_temps,
                    "r-o",
                    label="Hot-spot",
                    linewidth=2,
                )
            if any(topoil_temps):
                axes[1, 0].plot(
                    range(len(topoil_temps)),
                    topoil_temps,
                    "orange",
                    label="Top-oil",
                    linewidth=2,
                )
            axes[1, 0].axhline(
                y=110, color="red", linestyle="--", label="IEEE Limit (110°C)"
            )
            axes[1, 0].set_title("Temperature Trends")
            axes[1, 0].set_xlabel("Sample")
            axes[1, 0].set_ylabel("Temperature (°C)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Load trend
            loads = [d.get("load_percent", 0) for d in historical_data]
            if loads:
                axes[1, 1].plot(range(len(loads)), loads, "g-o", linewidth=2)
                axes[1, 1].axhline(
                    y=100, color="red", linestyle="--", label="Rated (100%)"
                )
                axes[1, 1].set_title("Load Trend")
                axes[1, 1].set_xlabel("Sample")
                axes[1, 1].set_ylabel("Load (%)")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
            plt.close()

            return temp_file.name

        except Exception as e:
            print(f"Error generating charts: {e}")
            return None

    def _generate_html_report(
        self,
        transformer_data: Dict[str, Any],
        dga_results: Optional[Dict[str, Any]],
        thermal_results: Optional[Dict[str, Any]],
        health_results: Optional[Dict[str, Any]],
        prediction_results: Optional[Dict[str, Any]],
        alerts: Optional[List[Dict[str, Any]]],
        recommendations: Optional[List[Dict[str, Any]]],
        historical_data: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Generate HTML report as fallback"""

        # Use HTML template if available, otherwise generate simple HTML
        template_path = os.path.join(
            os.path.dirname(__file__), "templates", "report_template.html"
        )

        if os.path.exists(template_path):
            # Use template
            try:
                from jinja2 import Template

                with open(template_path, "r") as f:
                    template = Template(f.read())

                html_content = template.render(
                    transformer=transformer_data,
                    dga=dga_results,
                    thermal=thermal_results,
                    health=health_results,
                    predictions=prediction_results,
                    alerts=alerts or [],
                    recommendations=recommendations or [],
                    report_date=datetime.now().strftime("%B %d, %Y"),
                )
            except ImportError:
                # Jinja2 not available, generate simple HTML
                html_content = self._generate_simple_html(
                    transformer_data,
                    dga_results,
                    thermal_results,
                    health_results,
                    prediction_results,
                    alerts,
                    recommendations,
                )
        else:
            # Generate simple HTML
            html_content = self._generate_simple_html(
                transformer_data,
                dga_results,
                thermal_results,
                health_results,
                prediction_results,
                alerts,
                recommendations,
            )

        # Save HTML
        output_path = self.config.output_path.replace(".pdf", ".html")
        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _generate_simple_html(
        self,
        transformer_data: Dict[str, Any],
        dga_results: Optional[Dict[str, Any]],
        thermal_results: Optional[Dict[str, Any]],
        health_results: Optional[Dict[str, Any]],
        prediction_results: Optional[Dict[str, Any]],
        alerts: Optional[List[Dict[str, Any]]],
        recommendations: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Generate simple HTML report"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Transformer Condition Assessment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #1a1a2e; text-align: center; }}
        h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 5px; }}
        h3 {{ color: #0f3460; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #0f3460; color: white; }}
        .alert {{ color: #c0392b; font-weight: bold; }}
        .warning {{ background-color: #fff3cd; }}
        .critical {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>Transformer Condition Assessment Report</h1>
    <p><strong>Transformer:</strong> {transformer_data.get("name", transformer_data.get("transformer_id", "Unknown"))}</p>
    <p><strong>Report Date:</strong> {datetime.now().strftime("%B %d, %Y")}</p>
"""

        # Executive Summary
        if health_results:
            html += f"""
    <h2>Executive Summary</h2>
    <p><strong>Overall Condition:</strong> {health_results.get("category", "Unknown")} (Health Index: {health_results.get("health_score", 0):.1f}/100)</p>
"""

        if prediction_results and prediction_results.get("rul_days"):
            html += f"""
    <p><strong>Estimated Remaining Life:</strong> {prediction_results["rul_days"]:.0f} days</p>
"""

        if alerts:
            critical = len([a for a in alerts if a.get("priority") == "CRITICAL"])
            high = len([a for a in alerts if a.get("priority") == "HIGH"])
            html += f"""
    <p class="alert">Active Alerts: {critical} Critical, {high} High Priority</p>
"""

        # DGA Analysis
        if dga_results:
            html += """
    <h2>DGA Analysis</h2>
"""
            if dga_results.get("fault_type"):
                html += f"""
    <p><strong>Primary Fault Type:</strong> {dga_results["fault_type"]}</p>
    <p><strong>Description:</strong> {dga_results.get("fault_description", "N/A")}</p>
"""

        # Thermal Analysis
        if thermal_results:
            html += """
    <h2>Thermal Analysis</h2>
"""
            if thermal_results.get("hotspot_temp"):
                html += f"""
    <p><strong>Hot-spot Temperature:</strong> {thermal_results["hotspot_temp"]:.1f}°C ({thermal_results.get("hotspot_status", "Normal")})</p>
"""

        # Health Index
        if health_results:
            html += """
    <h2>Health Index</h2>
"""
            html += f"""
    <p><strong>Health Score:</strong> {health_results.get("health_score", 0):.1f}/100</p>
    <p><strong>Category:</strong> {health_results.get("category", "Unknown")}</p>
"""

        # Recommendations
        if recommendations:
            html += """
    <h2>Maintenance Recommendations</h2>
"""
            for rec in recommendations:
                priority = rec.get("priority", 3)
                html += f"""
    <h3>{rec.get("action", "N/A")}</h3>
    <p><strong>Priority:</strong> {priority}</p>
    <p><strong>Timeframe:</strong> {rec.get("timeframe", "N/A")}</p>
    <p><strong>Rationale:</strong> {rec.get("rationale", "N/A")}</p>
"""

        # Alerts
        if alerts:
            html += """
    <h2>Active Alerts</h2>
"""
            for alert in alerts:
                html += f"""
    <p class="alert">[{alert.get("priority", "INFO")}] {alert.get("title", "N/A")}</p>
    <p>{alert.get("message", "N/A")}</p>
"""

        html += """
    <hr>
    <p><small>This report is generated in accordance with IEEE C57.104 and IEEE C57.91 standards.</small></p>
</body>
</html>
"""

        return html
