"""
Maintenance Recommendations Generator
Generates actionable maintenance recommendations based on:
- Fault type from DGA diagnosis
- Health index category
- Thermal analysis results
- RUL estimates
Includes IEEE standard references
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Recommendation:
    """Recommendation dataclass containing maintenance recommendation"""

    recommendation_id: str
    transformer_id: str
    transformer_name: str
    category: str
    priority: int  # 1-5 (1 = highest)
    action: str
    rationale: str
    timeframe: str
    estimated_cost: Optional[str] = None
    references: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "recommendation_id": self.recommendation_id,
            "transformer_id": self.transformer_id,
            "transformer_name": self.transformer_name,
            "category": self.category,
            "priority": self.priority,
            "action": self.action,
            "rationale": self.rationale,
            "timeframe": self.timeframe,
            "estimated_cost": self.estimated_cost,
            "references": self.references,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def get_priority_label(self) -> str:
        """Get human-readable priority label"""
        labels = {1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Informational"}
        return labels.get(self.priority, "Unknown")


class ActionRecommender:
    """
    Maintenance Recommendations Generator
    Maps fault types, health index, thermal analysis, and RUL to specific actions
    Includes IEEE standard references
    """

    # IEEE Standard references
    IEEE_STANDARDS = {
        "C57.104": "IEEE Guide for the Interpretation of Gases Generated in Oil-Immersed Transformers",
        "C57.91": "IEEE Guide for Loading Mineral-Oil-Immersed Transformers",
        "C57.12.00": "IEEE Standard General Requirements for Liquid-Immersed Distribution Transformers",
        "C57.19.03": "IEEE Standard for Bushings for DC Application",
        "C57.110": "IEEE Recommended Practice for Establishing Transformer Capability When Supplying Nonsinusoidal Load Currents",
        "C57.125": "IEEE Guide for Failure Investigation, Documentation, and Analysis for Power Transformers",
    }

    # Default recommendations by fault type
    FAULT_RECOMMENDATIONS = {
        "PD": {
            "actions": [
                {
                    "action": "Perform frequency response analysis (FRA)",
                    "timeframe": "Within 30 days",
                    "rationale": "PD activity may indicate mechanical deformation or insulation voids",
                    "reference": "IEEE C57.104",
                    "priority": 3,
                },
                {
                    "action": "Inspect bushings and high-voltage terminals",
                    "timeframe": "Within 14 days",
                    "rationale": "PD often originates from termination issues",
                    "reference": "IEEE C57.104",
                    "priority": 2,
                },
                {
                    "action": "Review loading history for overvoltage conditions",
                    "timeframe": "Within 7 days",
                    "rationale": "Overvoltage can accelerate PD activity",
                    "reference": "IEEE C57.91",
                    "priority": 3,
                },
            ]
        },
        "D1": {
            "actions": [
                {
                    "action": "Internal inspection of tank and windings",
                    "timeframe": "Within 14 days",
                    "rationale": "Low-energy discharge may indicate loose connections or minor arcing",
                    "reference": "IEEE C57.104",
                    "priority": 2,
                },
                {
                    "action": "Perform insulation resistance testing",
                    "timeframe": "Within 7 days",
                    "rationale": "Verify insulation integrity",
                    "reference": "IEEE C57.12.00",
                    "priority": 2,
                },
                {
                    "action": "Consider oil purification treatment",
                    "timeframe": "Within 30 days",
                    "rationale": "Remove dissolved gases and moisture",
                    "reference": "IEEE C57.104",
                    "priority": 3,
                },
            ]
        },
        "D2": {
            "actions": [
                {
                    "action": "Emergency internal inspection",
                    "timeframe": "Within 7 days",
                    "rationale": "High-energy discharge indicates severe fault activity - arcing",
                    "reference": "IEEE C57.104",
                    "priority": 1,
                },
                {
                    "action": "Comprehensive electrical testing",
                    "timeframe": "Within 7 days",
                    "rationale": "Assess extent of damage to insulation system",
                    "reference": "IEEE C57.12.00",
                    "priority": 1,
                },
                {
                    "action": "Prepare for transformer replacement evaluation",
                    "timeframe": "Immediate",
                    "rationale": "D2 faults often indicate irreversible damage",
                    "reference": "IEEE C57.125",
                    "priority": 1,
                },
                {
                    "action": "Oil analysis and dissolved gas trending",
                    "timeframe": "Within 3 days",
                    "rationale": "Monitor gas generation rate",
                    "reference": "IEEE C57.104",
                    "priority": 1,
                },
            ]
        },
        "T1": {
            "actions": [
                {
                    "action": "Cooling system inspection and cleaning",
                    "timeframe": "Within 30 days",
                    "rationale": "Low-temperature thermal fault may indicate cooling issues",
                    "reference": "IEEE C57.91",
                    "priority": 3,
                },
                {
                    "action": "Check load levels and distribution",
                    "timeframe": "Within 14 days",
                    "rationale": "Overloading can cause localized heating",
                    "reference": "IEEE C57.91",
                    "priority": 2,
                },
                {
                    "action": "Infrared thermography survey",
                    "timeframe": "Within 14 days",
                    "rationale": "Identify hot spots visually",
                    "reference": "IEEE C57.91",
                    "priority": 2,
                },
            ]
        },
        "T2": {
            "actions": [
                {
                    "action": "Reduce loading to 70% of rated capacity",
                    "timeframe": "Immediate (48 hours)",
                    "rationale": "Medium-temperature thermal fault (300-700°C) indicates significant degradation",
                    "reference": "IEEE C57.91",
                    "priority": 1,
                },
                {
                    "action": "Detailed cooling system inspection",
                    "timeframe": "Within 7 days",
                    "rationale": "Verify all cooling equipment operational",
                    "reference": "IEEE C57.91",
                    "priority": 1,
                },
                {
                    "action": "Internal inspection for hot spots",
                    "timeframe": "Within 14 days",
                    "rationale": "T2 faults indicate thermal deterioration",
                    "reference": "IEEE C57.104",
                    "priority": 2,
                },
                {
                    "action": "Evaluate for transformer replacement",
                    "timeframe": "Within 30 days",
                    "rationale": "T2 faults suggest advanced degradation",
                    "reference": "IEEE C57.125",
                    "priority": 2,
                },
            ]
        },
        "T3": {
            "actions": [
                {
                    "action": "Immediate load reduction to 50%",
                    "timeframe": "Immediate",
                    "rationale": "High-temperature thermal fault (>700°C) is critical - severe overheating",
                    "reference": "IEEE C57.91",
                    "priority": 1,
                },
                {
                    "action": "Emergency cooling system verification",
                    "timeframe": "Within 24 hours",
                    "rationale": "Verify all cooling methods operational",
                    "reference": "IEEE C57.91",
                    "priority": 1,
                },
                {
                    "action": "Schedule immediate internal inspection",
                    "timeframe": "Within 7 days",
                    "rationale": "T3 faults indicate severe thermal fault with carbonization risk",
                    "reference": "IEEE C57.104",
                    "priority": 1,
                },
                {
                    "action": "Prepare transformer replacement plan",
                    "timeframe": "Immediate",
                    "rationale": "T3 faults typically require transformer replacement",
                    "reference": "IEEE C57.125",
                    "priority": 1,
                },
            ]
        },
    }

    # Health-based recommendations
    HEALTH_RECOMMENDATIONS = {
        "critical": {
            "priority": 1,
            "actions": [
                {
                    "action": "Notify management of critical condition",
                    "timeframe": "Immediate",
                    "rationale": "Health index indicates transformer at end of life",
                    "reference": "IEEE C57.125",
                },
                {
                    "action": "Implement condition monitoring program",
                    "timeframe": "Immediate",
                    "rationale": "Increase monitoring frequency for critical assets",
                    "reference": "IEEE C57.104",
                },
                {
                    "action": "Develop replacement plan and budget",
                    "timeframe": "Within 30 days",
                    "rationale": "Plan for capital expenditure",
                    "reference": "IEEE C57.125",
                },
            ],
        },
        "poor": {
            "priority": 2,
            "actions": [
                {
                    "action": "Schedule detailed condition assessment",
                    "timeframe": "Within 30 days",
                    "rationale": "Poor health requires comprehensive evaluation",
                    "reference": "IEEE C57.104",
                },
                {
                    "action": "Review maintenance history and costs",
                    "timeframe": "Within 14 days",
                    "rationale": "Assess cost-effectiveness of continued maintenance",
                    "reference": "IEEE C57.125",
                },
                {
                    "action": "Evaluate spare transformer availability",
                    "timeframe": "Within 60 days",
                    "rationale": "Plan for contingency",
                    "reference": "IEEE C57.12.00",
                },
            ],
        },
        "fair": {
            "priority": 3,
            "actions": [
                {
                    "action": "Schedule routine maintenance",
                    "timeframe": "Within 90 days",
                    "rationale": "Fair condition - maintain through scheduled maintenance",
                    "reference": "IEEE C57.91",
                },
                {
                    "action": "Continue DGA monitoring program",
                    "timeframe": "Ongoing",
                    "regulare": "Track trends for early fault detection",
                    "reference": "IEEE C57.104",
                },
            ],
        },
    }

    # Thermal-based recommendations
    THERMAL_RECOMMENDATIONS = {
        "hotspot_high": {
            "actions": [
                {
                    "action": "Reduce load to recommended level",
                    "timeframe": "Within 48 hours",
                    "rationale": "Hot-spot temperature above IEEE C57.91 alarm limit",
                    "reference": "IEEE C57.91",
                    "priority": 1,
                },
                {
                    "action": "Verify cooling system operation",
                    "timeframe": "Within 24 hours",
                    "rationale": "Check fans, pumps, and radiators",
                    "reference": "IEEE C57.91",
                    "priority": 1,
                },
                {
                    "action": "Inspect oil level and quality",
                    "timeframe": "Within 7 days",
                    "rationale": "Low oil or degradation can cause overheating",
                    "reference": "IEEE C57.104",
                    "priority": 2,
                },
            ]
        },
        "topoil_high": {
            "actions": [
                {
                    "action": "Monitor top-oil temperature trend",
                    "timeframe": "Ongoing",
                    "rationale": "High top-oil temperature indicates cooling issues",
                    "reference": "IEEE C57.91",
                    "priority": 2,
                },
                {
                    "action": "Clean radiators and heat exchangers",
                    "timeframe": "Within 30 days",
                    "rationale": "Improve cooling efficiency",
                    "reference": "IEEE C57.91",
                    "priority": 3,
                },
            ]
        },
    }

    # RUL-based recommendations
    RUL_RECOMMENDATIONS = {
        "critical": {  # < 30 days
            "priority": 1,
            "actions": [
                {
                    "action": "Emergency transformer replacement planning",
                    "timeframe": "Immediate",
                    "rationale": "RUL less than 30 days - critical failure risk",
                    "reference": "IEEE C57.125",
                },
                {
                    "action": "Implement risk mitigation measures",
                    "timeframe": "Immediate",
                    "rationale": "Reduce load and increase monitoring",
                    "reference": "IEEE C57.91",
                },
                {
                    "action": "Notify all stakeholders",
                    "timeframe": "Immediate",
                    "rationale": "Ensure all parties aware of critical condition",
                    "reference": "IEEE C57.125",
                },
            ],
        },
        "warning": {  # 30-90 days
            "priority": 2,
            "actions": [
                {
                    "action": "Expedite replacement procurement",
                    "timeframe": "Within 30 days",
                    "rationale": "RUL indicates replacement needed soon",
                    "reference": "IEEE C57.125",
                },
                {
                    "action": "Increase monitoring frequency",
                    "timeframe": "Immediate",
                    "rationale": "Track condition more closely",
                    "reference": "IEEE C57.104",
                },
            ],
        },
        "monitor": {  # 90-180 days
            "priority": 3,
            "actions": [
                {
                    "action": "Include in next budget cycle",
                    "timeframe": "Within 90 days",
                    "rationale": "Plan for replacement in next fiscal year",
                    "reference": "IEEE C57.125",
                },
                {
                    "action": "Continue routine monitoring",
                    "timeframe": "Ongoing",
                    "rationale": "Track RUL trend",
                    "reference": "IEEE C57.104",
                },
            ],
        },
    }

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the action recommender

        Args:
            template_dir: Directory containing recommendation templates
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.templates = self._load_templates()

    def _get_default_template_dir(self) -> str:
        """Get default template directory"""
        return os.path.join(os.path.dirname(__file__), "templates")

    def _load_templates(self) -> Dict[str, Any]:
        """Load recommendation templates from YAML file"""
        templates = {}
        template_file = os.path.join(self.template_dir, "recommendation_templates.yaml")

        if os.path.exists(template_file):
            with open(template_file, "r") as f:
                templates = yaml.safe_load(f) or {}

        return templates

    def generate_recommendations(
        self,
        transformer_id: str,
        transformer_name: str,
        health_score: Optional[float] = None,
        health_category: Optional[str] = None,
        fault_type: Optional[str] = None,
        hotspot_temp: Optional[float] = None,
        topoil_temp: Optional[float] = None,
        rul_days: Optional[float] = None,
        failure_probability: Optional[float] = None,
        gas_concentrations: Optional[Dict[str, float]] = None,
    ) -> List[Recommendation]:
        """
        Generate maintenance recommendations

        Args:
            transformer_id: Unique transformer identifier
            transformer_name: Transformer name
            health_score: Health index (0-100)
            health_category: Health category (Critical, Poor, Fair, Good, Excellent)
            fault_type: DGA fault type (PD, D1, D2, T1, T2, T3)
            hotspot_temp: Hot-spot temperature (°C)
            topoil_temp: Top-oil temperature (°C)
            rul_days: Remaining useful life (days)
            failure_probability: Failure probability (0-1)
            gas_concentrations: DGA gas concentrations

        Returns:
            List of recommendations
        """
        recommendations = []

        # Get health category if not provided
        if health_category is None and health_score is not None:
            health_category = self._get_health_category(health_score)

        # Generate fault-based recommendations
        if fault_type and fault_type in self.FAULT_RECOMMENDATIONS:
            fault_recs = self.FAULT_RECOMMENDATIONS[fault_type]
            for rec in fault_recs["actions"]:
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    category=f"DGA - {fault_type}",
                    priority=rec.get("priority", 3),
                    action=rec["action"],
                    rationale=rec["rationale"],
                    timeframe=rec["timeframe"],
                    references=[
                        self.IEEE_STANDARDS.get(rec["reference"], rec["reference"])
                    ],
                    metadata={"fault_type": fault_type},
                )
                recommendations.append(recommendation)

        # Generate health-based recommendations
        if health_category and health_category.lower() in self.HEALTH_RECOMMENDATIONS:
            health_recs = self.HEALTH_RECOMMENDATIONS[health_category.lower()]
            for rec in health_recs["actions"]:
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    category=f"Health - {health_category}",
                    priority=health_recs["priority"],
                    action=rec["action"],
                    rationale=rec["rationale"],
                    timeframe=rec["timeframe"],
                    references=[
                        self.IEEE_STANDARDS.get(rec["reference"], rec["reference"])
                    ],
                    metadata={
                        "health_category": health_category,
                        "health_score": health_score,
                    },
                )
                recommendations.append(recommendation)

        # Generate thermal-based recommendations
        if hotspot_temp is not None and hotspot_temp > 120:
            thermal_recs = self.THERMAL_RECOMMENDATIONS["hotspot_high"]
            for rec in thermal_recs["actions"]:
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    category="Thermal - Hot-spot",
                    priority=rec.get("priority", 2),
                    action=rec["action"],
                    rationale=rec["rationale"],
                    timeframe=rec["timeframe"],
                    references=[
                        self.IEEE_STANDARDS.get(rec["reference"], rec["reference"])
                    ],
                    metadata={"hotspot_temp": hotspot_temp},
                )
                recommendations.append(recommendation)

        if topoil_temp is not None and topoil_temp > 105:
            thermal_recs = self.THERMAL_RECOMMENDATIONS["topoil_high"]
            for rec in thermal_recs["actions"]:
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    category="Thermal - Top-oil",
                    priority=rec.get("priority", 3),
                    action=rec["action"],
                    rationale=rec["rationale"],
                    timeframe=rec["timeframe"],
                    references=[
                        self.IEEE_STANDARDS.get(rec["reference"], rec["reference"])
                    ],
                    metadata={"topoil_temp": topoil_temp},
                )
                recommendations.append(recommendation)

        # Generate RUL-based recommendations
        if rul_days is not None:
            if rul_days < 30:
                rul_category = "critical"
            elif rul_days < 90:
                rul_category = "warning"
            elif rul_days < 180:
                rul_category = "monitor"
            else:
                rul_category = None

            if rul_category and rul_category in self.RUL_RECOMMENDATIONS:
                rul_recs = self.RUL_RECOMMENDATIONS[rul_category]
                for rec in rul_recs["actions"]:
                    recommendation = Recommendation(
                        recommendation_id=str(uuid.uuid4()),
                        transformer_id=transformer_id,
                        transformer_name=transformer_name,
                        category="RUL Prediction",
                        priority=rul_recs["priority"],
                        action=rec["action"],
                        rationale=rec["rationale"],
                        timeframe=rec["timeframe"],
                        references=[
                            self.IEEE_STANDARDS.get(rec["reference"], rec["reference"])
                        ],
                        metadata={"rul_days": rul_days},
                    )
                    recommendations.append(recommendation)

        # Add cost estimates for critical recommendations
        for rec in recommendations:
            if rec.priority == 1:
                rec.estimated_cost = self._estimate_cost(rec.category, rec.action)

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations

    def _get_health_category(self, health_score: float) -> str:
        """Get health category from health score"""
        if health_score < 25:
            return "Critical"
        elif health_score < 50:
            return "Poor"
        elif health_score < 70:
            return "Fair"
        elif health_score < 85:
            return "Good"
        else:
            return "Excellent"

    def _estimate_cost(self, category: str, action: str) -> str:
        """Estimate cost for recommendation"""
        # Simple cost estimation based on action type
        action_lower = action.lower()

        if "replacement" in action_lower or "procurement" in action_lower:
            return "$150,000 - $500,000+"
        elif "internal inspection" in action_lower:
            return "$10,000 - $30,000"
        elif "oil" in action_lower or "purification" in action_lower:
            return "$5,000 - $15,000"
        elif "cooling" in action_lower or "clean" in action_lower:
            return "$2,000 - $10,000"
        elif "testing" in action_lower or "analysis" in action_lower:
            return "$3,000 - $15,000"
        elif "monitoring" in action_lower:
            return "$500 - $2,000"
        else:
            return "Varies"

    def get_recommendation_summary(
        self, recommendations: List[Recommendation]
    ) -> Dict[str, Any]:
        """
        Get summary of recommendations

        Args:
            recommendations: List of recommendations

        Returns:
            Summary dictionary
        """
        summary = {
            "total": len(recommendations),
            "by_priority": {
                "Critical (1)": len([r for r in recommendations if r.priority == 1]),
                "High (2)": len([r for r in recommendations if r.priority == 2]),
                "Medium (3)": len([r for r in recommendations if r.priority == 3]),
                "Low (4)": len([r for r in recommendations if r.priority == 4]),
                "Info (5)": len([r for r in recommendations if r.priority == 5]),
            },
            "by_category": {},
            "immediate_actions": [r.action for r in recommendations if r.priority == 1],
            "recommendations": [r.to_dict() for r in recommendations],
        }

        # Count by category
        for rec in recommendations:
            if rec.category not in summary["by_category"]:
                summary["by_category"][rec.category] = 0
            summary["by_category"][rec.category] += 1

        return summary
