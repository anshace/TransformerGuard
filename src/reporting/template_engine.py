"""
Rule-Based Natural Language Engine for Transformer Diagnostics
Generates standards-based explanations using deterministic templates
Grounded in IEEE standards (C57.104, C57.91)
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TemplateResult:
    """Result from template engine containing generated explanation"""

    explanation: str
    sections: Dict[str, Any]
    generated_at: datetime
    transformer_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "explanation": self.explanation,
            "sections": self.sections,
            "generated_at": self.generated_at.isoformat(),
            "transformer_id": self.transformer_id,
        }


class TemplateEngine:
    """
    Rule-Based Natural Language Engine
    Generates deterministic, traceable, auditable explanations based on IEEE standards
    """

    # IEEE C57.104 gas thresholds (in ppm)
    GAS_THRESHOLDS = {
        "h2": 100,
        "ch4": 120,
        "c2h2": 2,
        "c2h4": 50,
        "c2h6": 65,
        "co": 350,
        "co2": 2500,
    }

    # IEEE C57.91 temperature limits
    TEMP_LIMITS = {
        "hotspot_normal": 110,
        "hotspot_emergency": 140,
        "hotspot_alarm": 120,
        "topoil_normal": 105,
        "topoil_emergency": 115,
    }

    # Duval Triangle fault types
    FAULT_DESCRIPTIONS = {
        "PD": "Partial Discharge",
        "D1": "Low-energy discharge",
        "D2": "High-energy discharge",
        "T1": "Thermal fault < 300°C",
        "T2": "Thermal fault 300-700°C",
        "T3": "Thermal fault > 700°C",
    }

    # Health score categories
    HEALTH_CATEGORIES = {
        (0, 25): "Critical",
        (25, 50): "Poor",
        (50, 70): "Fair",
        (70, 85): "Good",
        (85, 100): "Excellent",
    }

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the template engine

        Args:
            template_dir: Directory containing template files (optional)
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.templates = self._load_templates()

    def _get_default_template_dir(self) -> str:
        """Get default template directory"""
        return os.path.join(os.path.dirname(__file__), "templates")

    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from YAML files"""
        templates = {}

        # Load explanation templates if available
        template_file = os.path.join(self.template_dir, "explanation_templates.yaml")
        if os.path.exists(template_file):
            with open(template_file, "r") as f:
                templates = yaml.safe_load(f) or {}

        return templates

    def generate_explanation(
        self,
        transformer_id: str,
        transformer_name: str,
        health_score: float,
        duval_result: Optional[str] = None,
        doernenburg_result: Optional[Dict[str, Any]] = None,
        iec_result: Optional[Dict[str, Any]] = None,
        rogers_result: Optional[Dict[str, Any]] = None,
        key_gas_result: Optional[Dict[str, Any]] = None,
        hotspot_temp: Optional[float] = None,
        topoil_temp: Optional[float] = None,
        load_percent: Optional[float] = None,
        rul_days: Optional[float] = None,
        failure_probability: Optional[float] = None,
        gas_concentrations: Optional[Dict[str, float]] = None,
        trend_data: Optional[Dict[str, Any]] = None,
    ) -> TemplateResult:
        """
        Generate comprehensive natural language explanation

        Args:
            transformer_id: Unique transformer identifier
            transformer_name: Transformer name
            health_score: Health index (0-100)
            duval_result: Duval Triangle result
            doernenburg_result: Doernenburg ratios result
            iec_result: IEC ratios result
            rogers_result: Rogers ratios result
            key_gas_result: Key gas analysis result
            hotspot_temp: Hot-spot temperature (°C)
            topoil_temp: Top-oil temperature (°C)
            load_percent: Load percentage
            rul_days: Remaining useful life (days)
            failure_probability: Failure probability (0-1)
            gas_concentrations: DGA gas concentrations (ppm)
            trend_data: Historical trend data

        Returns:
            TemplateResult with explanation and sections
        """
        sections = {}

        # Generate diagnosis section
        sections["diagnosis"] = self._generate_diagnosis_section(
            duval_result,
            doernenburg_result,
            iec_result,
            rogers_result,
            key_gas_result,
            gas_concentrations,
        )

        # Generate thermal section
        sections["thermal"] = self._generate_thermal_section(
            hotspot_temp, topoil_temp, load_percent
        )

        # Generate health section
        sections["health"] = self._generate_health_section(
            health_score, gas_concentrations
        )

        # Generate predictions section
        sections["predictions"] = self._generate_predictions_section(
            rul_days, failure_probability, trend_data
        )

        # Generate actions section
        sections["actions"] = self._generate_actions_section(
            health_score, duval_result, hotspot_temp, rul_days
        )

        # Build full explanation
        explanation = self._build_explanation(
            transformer_name, sections, gas_concentrations
        )

        return TemplateResult(
            explanation=explanation,
            sections=sections,
            generated_at=datetime.utcnow(),
            transformer_id=transformer_id,
        )

    def _generate_diagnosis_section(
        self,
        duval_result: Optional[str],
        doernenburg_result: Optional[Dict[str, Any]],
        iec_result: Optional[Dict[str, Any]],
        rogers_result: Optional[Dict[str, Any]],
        key_gas_result: Optional[Dict[str, Any]],
        gas_concentrations: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Generate DGA diagnosis section"""
        section = {
            "primary_fault": None,
            "fault_description": None,
            "methods": {},
            "key_gases": [],
            "risk_level": "Normal",
        }

        # Determine primary fault from Duval Triangle
        if duval_result:
            section["primary_fault"] = duval_result
            section["fault_description"] = self.FAULT_DESCRIPTIONS.get(
                duval_result, "Unknown fault"
            )

            # Determine risk level based on fault type
            if duval_result in ["D2", "T3"]:
                section["risk_level"] = "High"
            elif duval_result in ["D1", "T2"]:
                section["risk_level"] = "Medium"
            elif duval_result == "PD":
                section["risk_level"] = "Low"

        # Add method results
        if doernenburg_result:
            section["methods"]["doernenburg"] = doernenburg_result
        if iec_result:
            section["methods"]["iec"] = iec_result
        if rogers_result:
            section["methods"]["rogers"] = rogers_result
        if key_gas_result:
            section["methods"]["key_gas"] = key_gas_result

        # Identify key elevated gases
        if gas_concentrations:
            elevated = []
            for gas, threshold in self.GAS_THRESHOLDS.items():
                if gas in gas_concentrations:
                    conc = gas_concentrations[gas]
                    if conc > threshold * 0.5:  # Report if above 50% of threshold
                        elevated.append(
                            {
                                "gas": gas.upper(),
                                "concentration": conc,
                                "threshold": threshold,
                                "exceeds_threshold": conc > threshold,
                            }
                        )
            section["key_gases"] = elevated

        return section

    def _generate_thermal_section(
        self,
        hotspot_temp: Optional[float],
        topoil_temp: Optional[float],
        load_percent: Optional[float],
    ) -> Dict[str, Any]:
        """Generate thermal analysis section"""
        section = {
            "hotspot_status": "Normal",
            "topoil_status": "Normal",
            "load_status": "Normal",
            "hotspot_temp": hotspot_temp,
            "topoil_temp": topoil_temp,
            "load_percent": load_percent,
            "warnings": [],
        }

        if hotspot_temp is not None:
            if hotspot_temp > self.TEMP_LIMITS["hotspot_emergency"]:
                section["hotspot_status"] = "Emergency"
                section["warnings"].append(
                    f"Hot-spot temperature {hotspot_temp}°C exceeds emergency limit "
                    f"of {self.TEMP_LIMITS['hotspot_emergency']}°C (IEEE C57.91)"
                )
            elif hotspot_temp > self.TEMP_LIMITS["hotspot_alarm"]:
                section["hotspot_status"] = "Alarm"
                section["warnings"].append(
                    f"Hot-spot temperature {hotspot_temp}°C exceeds alarm limit "
                    f"of {self.TEMP_LIMITS['hotspot_alarm']}°C"
                )

        if topoil_temp is not None:
            if topoil_temp > self.TEMP_LIMITS["topoil_emergency"]:
                section["topoil_status"] = "Emergency"
                section["warnings"].append(
                    f"Top-oil temperature {topoil_temp}°C exceeds emergency limit"
                )

        if load_percent is not None:
            if load_percent > 100:
                section["load_status"] = "Overload"
                section["warnings"].append(
                    f"Loading at {load_percent}% exceeds rated capacity"
                )
            elif load_percent > 80:
                section["load_status"] = "High"

        return section

    def _generate_health_section(
        self,
        health_score: float,
        gas_concentrations: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Generate health index section"""
        # Determine health category
        health_category = "Unknown"
        for (low, high), category in self.HEALTH_CATEGORIES.items():
            if low <= health_score < high:
                health_category = category
                break
        if health_score >= 85:
            health_category = "Excellent"

        section = {
            "health_score": health_score,
            "category": health_category,
            "condition": self._get_condition_description(health_score),
            "dga_status": "Normal",
        }

        # Check DGA status
        if gas_concentrations:
            exceed_count = 0
            for gas, threshold in self.GAS_THRESHOLDS.items():
                if gas in gas_concentrations:
                    if gas_concentrations[gas] > threshold:
                        exceed_count += 1

            if exceed_count >= 3:
                section["dga_status"] = "Critical"
            elif exceed_count >= 2:
                section["dga_status"] = "Warning"
            elif exceed_count >= 1:
                section["dga_status"] = "Monitor"

        return section

    def _get_condition_description(self, health_score: float) -> str:
        """Get condition description based on health score"""
        if health_score < 25:
            return "Critical condition - immediate action required"
        elif health_score < 50:
            return "Poor condition - urgent maintenance recommended"
        elif health_score < 70:
            return "Fair condition - scheduled maintenance required"
        elif health_score < 85:
            return "Good condition - routine monitoring"
        else:
            return "Excellent condition - normal operation"

    def _generate_predictions_section(
        self,
        rul_days: Optional[float],
        failure_probability: Optional[float],
        trend_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate predictions section"""
        section = {
            "rul_days": rul_days,
            "failure_probability": failure_probability,
            "rul_status": "Unknown",
            "trend": "Stable",
        }

        if rul_days is not None:
            if rul_days < 30:
                section["rul_status"] = "Critical"
            elif rul_days < 90:
                section["rul_status"] = "Warning"
            elif rul_days < 180:
                section["rul_status"] = "Monitor"
            else:
                section["rul_status"] = "Acceptable"

        if failure_probability is not None:
            if failure_probability > 0.7:
                section["risk_level"] = "High"
            elif failure_probability > 0.3:
                section["risk_level"] = "Medium"
            else:
                section["risk_level"] = "Low"

        if trend_data:
            section["trend"] = trend_data.get("overall_trend", "Stable")

        return section

    def _generate_actions_section(
        self,
        health_score: float,
        duval_result: Optional[str],
        hotspot_temp: Optional[float],
        rul_days: Optional[float],
    ) -> Dict[str, Any]:
        """Generate recommended actions section"""
        section = {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "urgency": "Normal",
        }

        # Determine urgency based on multiple factors
        if health_score < 25 or rul_days is not None and rul_days < 30:
            section["urgency"] = "Critical"
            section["immediate_actions"].append(
                "Reduce load immediately to minimize stress"
            )
            section["short_term_actions"].extend(
                [
                    "Schedule emergency inspection",
                    "Prepare contingency plan",
                    "Notify management of critical condition",
                ]
            )
        elif health_score < 50:
            section["urgency"] = "High"
            section["immediate_actions"].append("Monitor condition closely")
            section["short_term_actions"].extend(
                [
                    "Schedule detailed inspection within 7 days",
                    "Review maintenance history",
                ]
            )
        else:
            section["immediate_actions"].append("Continue routine monitoring")

        # Fault-specific actions
        if duval_result in ["D1", "D2"]:
            section["short_term_actions"].extend(
                [
                    "Perform internal inspection",
                    "Test insulation resistance",
                    "Consider oil purification",
                ]
            )
        elif duval_result in ["T2", "T3"]:
            section["short_term_actions"].extend(
                [
                    "Inspect cooling system",
                    "Check for hot spots",
                    "Verify load reduction effectiveness",
                ]
            )

        # Thermal-specific actions
        if hotspot_temp is not None and hotspot_temp > 120:
            recommended_load = max(50, 100 - (hotspot_temp - 120) * 2)
            section["immediate_actions"].append(
                f"Reduce loading to {recommended_load:.0f}% within 48 hours"
            )

        # Long-term actions
        if health_score < 70:
            section["long_term_actions"].extend(
                [
                    "Plan for transformer replacement",
                    "Budget for capital expenditure",
                    "Evaluate spare transformer availability",
                ]
            )

        return section

    def _build_explanation(
        self,
        transformer_name: str,
        sections: Dict[str, Any],
        gas_concentrations: Optional[Dict[str, float]],
    ) -> str:
        """Build full natural language explanation"""
        lines = []

        # Header
        lines.append(f"DIAGNOSTIC REPORT: {transformer_name}")
        lines.append("=" * 60)
        lines.append("")

        # Executive Summary
        health = sections["health"]
        diagnosis = sections["diagnosis"]

        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 30)

        if health["category"] == "Critical":
            lines.append(
                f"ALERT: Transformer {transformer_name} requires immediate attention. "
                f"Health score is {health['health_score']:.1f} ({health['category']})."
            )
        elif health["category"] == "Poor":
            lines.append(
                f"WARNING: Transformer {transformer_name} shows degradation. "
                f"Health score is {health['health_score']:.1f} ({health['category']})."
            )
        else:
            lines.append(
                f"Transformer {transformer_name} condition: {health['category']} "
                f"(Health Score: {health['health_score']:.1f})."
            )

        if diagnosis["primary_fault"]:
            lines.append(
                f"Primary fault indication: {diagnosis['fault_description']} "
                f"({diagnosis['primary_fault']}) - Risk Level: {diagnosis['risk_level']}."
            )

        lines.append("")

        # DGA Analysis
        lines.append("DGA ANALYSIS")
        lines.append("-" * 30)

        if diagnosis["key_gases"]:
            lines.append("Key gas concentrations:")
            for kg in diagnosis["key_gases"]:
                status = "EXCEEDS" if kg["exceeds_threshold"] else "Elevated"
                lines.append(
                    f"  - {kg['gas']}: {kg['concentration']:.1f} ppm "
                    f"(Threshold: {kg['threshold']} ppm) - {status}"
                )
        else:
            lines.append("All gas concentrations within normal limits.")

        lines.append("")

        # Thermal Analysis
        thermal = sections["thermal"]
        lines.append("THERMAL ANALYSIS")
        lines.append("-" * 30)

        if thermal["hotspot_temp"] is not None:
            lines.append(
                f"Hot-spot Temperature: {thermal['hotspot_temp']:.1f}°C "
                f"(Status: {thermal['hotspot_status']}, "
                f"IEEE C57.91 Limit: {self.TEMP_LIMITS['hotspot_normal']}°C normal, "
                f"{self.TEMP_LIMITS['hotspot_emergency']}°C emergency)"
            )

        if thermal["topoil_temp"] is not None:
            lines.append(
                f"Top-Oil Temperature: {thermal['topoil_temp']:.1f}°C "
                f"(Status: {thermal['topoil_status']})"
            )

        if thermal["load_percent"] is not None:
            lines.append(
                f"Loading: {thermal['load_percent']:.1f}% of rated capacity "
                f"(Status: {thermal['load_status']})"
            )

        if thermal["warnings"]:
            lines.append("Warnings:")
            for warning in thermal["warnings"]:
                lines.append(f"  - {warning}")

        lines.append("")

        # Health Assessment
        lines.append("HEALTH ASSESSMENT")
        lines.append("-" * 30)
        lines.append(f"Health Index: {health['health_score']:.1f}/100")
        lines.append(f"Category: {health['category']}")
        lines.append(f"Condition: {health['condition']}")
        lines.append(f"DGA Status: {health['dga_status']}")
        lines.append("")

        # Predictions
        predictions = sections["predictions"]
        lines.append("PREDICTIONS & FORECASTS")
        lines.append("-" * 30)

        if predictions["rul_days"] is not None:
            lines.append(
                f"Estimated Remaining Useful Life: {predictions['rul_days']:.0f} days "
                f"(Status: {predictions['rul_status']})"
            )

        if predictions["failure_probability"] is not None:
            prob_pct = predictions["failure_probability"] * 100
            lines.append(
                f"Probability of Failure (12 months): {prob_pct:.1f}% "
                f"(Risk Level: {predictions.get('risk_level', 'Unknown')})"
            )

        lines.append("")

        # Recommendations
        actions = sections["actions"]
        lines.append("RECOMMENDED ACTIONS")
        lines.append("-" * 30)
        lines.append(f"Urgency Level: {actions['urgency']}")
        lines.append("")

        if actions["immediate_actions"]:
            lines.append("IMMEDIATE ACTIONS (Within 24-48 hours):")
            for action in actions["immediate_actions"]:
                lines.append(f"  • {action}")
            lines.append("")

        if actions["short_term_actions"]:
            lines.append("SHORT-TERM ACTIONS (Within 1-2 weeks):")
            for action in actions["short_term_actions"]:
                lines.append(f"  • {action}")
            lines.append("")

        if actions["long_term_actions"]:
            lines.append("LONG-TERM ACTIONS (Planning horizon):")
            for action in actions["long_term_actions"]:
                lines.append(f"  • {action}")
            lines.append("")

        # Footer
        lines.append("=" * 60)
        lines.append("Report generated in accordance with IEEE C57.104 and IEEE C57.91")

        return "\n".join(lines)

    def generate_fault_explanation(
        self,
        fault_type: str,
        transformer_name: str,
        gas_concentrations: Dict[str, float],
        hotspot_temp: Optional[float] = None,
        rul_days: Optional[float] = None,
    ) -> str:
        """
        Generate focused explanation for specific fault type

        Args:
            fault_type: Fault type code (D1, D2, T1, T2, T3, PD)
            transformer_name: Transformer name
            gas_concentrations: Gas concentrations in ppm
            hotspot_temp: Hot-spot temperature
            rul_days: Remaining useful life

        Returns:
            Fault-specific explanation string
        """
        lines = []

        lines.append(
            f"ALERT: Transformer {transformer_name} shows {self.FAULT_DESCRIPTIONS.get(fault_type, 'anomaly')}."
        )

        # Fault-specific details
        if fault_type in ["D1", "D2"]:
            # Discharge faults
            lines.append("")
            lines.append("DGA Analysis:")
            if "c2h2" in gas_concentrations:
                lines.append(
                    f"  - Acetylene (C2H2): {gas_concentrations['c2h2']:.1f} ppm "
                    f"(threshold: {self.GAS_THRESHOLDS['c2h2']} ppm)"
                )
            if "h2" in gas_concentrations:
                lines.append(
                    f"  - Hydrogen (H2): {gas_concentrations['h2']:.1f} ppm "
                    f"(threshold: {self.GAS_THRESHOLDS['h2']} ppm)"
                )

            lines.append("")
            lines.append("DIAGNOSIS:")
            if fault_type == "D2":
                lines.append("High-energy discharge detected (arcing).")
                lines.append("This indicates severe internal fault activity.")
                lines.append(
                    "Recommend immediate load reduction and emergency inspection."
                )
            else:
                lines.append("Low-energy discharge detected (sparking).")
                lines.append(
                    "May indicate loose connections or insulation degradation."
                )

        elif fault_type in ["T1", "T2", "T3"]:
            # Thermal faults
            lines.append("")
            lines.append("DGA Analysis:")
            if "c2h4" in gas_concentrations:
                lines.append(
                    f"  - Ethylene (C2H4): {gas_concentrations['c2h4']:.1f} ppm "
                    f"(threshold: {self.GAS_THRESHOLDS['c2h4']} ppm)"
                )
            if "ch4" in gas_concentrations:
                lines.append(
                    f"  - Methane (CH4): {gas_concentrations['ch4']:.1f} ppm "
                    f"(threshold: {self.GAS_THRESHOLDS['ch4']} ppm)"
                )

            lines.append("")
            lines.append("DIAGNOSIS:")
            if fault_type == "T1":
                lines.append("Low-temperature thermal fault detected (<300°C).")
                lines.append("May indicate localized overheating.")
            elif fault_type == "T2":
                lines.append("Medium-temperature thermal fault detected (300-700°C).")
                lines.append("Indicates significant thermal degradation.")
            else:
                lines.append("High-temperature thermal fault detected (>700°C).")
                lines.append("Critical condition - severe overheating detected.")

        elif fault_type == "PD":
            lines.append("")
            lines.append("DGA Analysis:")
            if "h2" in gas_concentrations:
                lines.append(
                    f"  - Hydrogen (H2): {gas_concentrations['h2']:.1f} ppm "
                    f"(threshold: {self.GAS_THRESHOLDS['h2']} ppm)"
                )

            lines.append("")
            lines.append("DIAGNOSIS:")
            lines.append("Partial discharge activity detected.")
            lines.append("May indicate insulation deterioration or void formation.")

        # Thermal details
        if hotspot_temp is not None:
            lines.append("")
            lines.append(f"Hot-spot Temperature: {hotspot_temp:.1f}°C")
            if hotspot_temp > self.TEMP_LIMITS["hotspot_emergency"]:
                lines.append(
                    f"EXCEEDS IEEE C57.91 emergency limit ({self.TEMP_LIMITS['hotspot_emergency']}°C)"
                )
            elif hotspot_temp > self.TEMP_LIMITS["hotspot_normal"]:
                lines.append(
                    f"EXCEEDS IEEE C57.91 normal limit ({self.TEMP_LIMITS['hotspot_normal']}°C)"
                )

        # RUL
        if rul_days is not None:
            lines.append("")
            lines.append(f"Estimated Remaining Useful Life: {rul_days:.0f} days")
            if rul_days < 30:
                lines.append("CRITICAL: Immediate action required.")
            elif rul_days < 90:
                lines.append("URGENT: Schedule maintenance within 2 weeks.")

        return "\n".join(lines)
