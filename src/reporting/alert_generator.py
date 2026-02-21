"""
Priority-Ranked Alert Generator
Generates alerts with priority levels based on transformer condition
Supports alert acknowledgment and unique alert IDs
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml


class AlertPriority(Enum):
    """Alert priority levels"""

    CRITICAL = "CRITICAL"  # Immediate action required
    HIGH = "HIGH"  # Action within 24-48 hours
    MEDIUM = "MEDIUM"  # Action within 1 week
    LOW = "LOW"  # Monitor closely
    INFO = "INFO"  # Informational


class AlertCategory(Enum):
    """Alert categories"""

    DGA = "DGA"
    THERMAL = "THERMAL"
    HEALTH = "HEALTH"
    LOADING = "LOADING"
    ANOMALY = "ANOMALY"
    PREDICTION = "PREDICTION"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class Alert:
    """Alert dataclass containing all alert information"""

    alert_id: str
    transformer_id: str
    transformer_name: str
    priority: AlertPriority
    category: AlertCategory
    title: str
    message: str
    created_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "alert_id": self.alert_id,
            "transformer_id": self.transformer_id,
            "transformer_name": self.transformer_name,
            "priority": self.priority.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat()
            if self.acknowledged_at
            else None,
            "acknowledged_by": self.acknowledged_by,
            "actions": self.actions,
            "metadata": self.metadata,
        }

    def acknowledge(self, acknowledged_by: str = "System") -> None:
        """Mark alert as acknowledged"""
        self.acknowledged = True
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = acknowledged_by

    def get_priority_order(self) -> int:
        """Get numeric priority order for sorting"""
        priority_map = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3,
            AlertPriority.INFO: 4,
        }
        return priority_map.get(self.priority, 5)


class AlertGenerator:
    """
    Priority-Ranked Alert Generator
    Generates alerts based on transformer condition data
    """

    # Alert thresholds based on IEEE standards
    DGA_THRESHOLDS = {
        "h2": 100,
        "ch4": 120,
        "c2h2": 2,
        "c2h4": 50,
        "c2h6": 65,
        "co": 350,
        "co2": 2500,
    }

    THERMAL_THRESHOLDS = {
        "hotspot_alarm": 120,
        "hotspot_emergency": 140,
        "topoil_alarm": 105,
        "topoil_emergency": 115,
    }

    HEALTH_THRESHOLDS = {
        "critical": 25,
        "poor": 50,
        "fair": 70,
    }

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the alert generator

        Args:
            template_dir: Directory containing alert templates
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.templates = self._load_templates()
        self.active_alerts: Dict[str, Alert] = {}

    def _get_default_template_dir(self) -> str:
        """Get default template directory"""
        return os.path.join(os.path.dirname(__file__), "templates")

    def _load_templates(self) -> Dict[str, Any]:
        """Load alert templates from YAML file"""
        templates = {}
        template_file = os.path.join(self.template_dir, "alert_templates.yaml")

        if os.path.exists(template_file):
            with open(template_file, "r") as f:
                templates = yaml.safe_load(f) or {}

        return templates

    def generate_alerts(
        self,
        transformer_id: str,
        transformer_name: str,
        health_score: Optional[float] = None,
        fault_type: Optional[str] = None,
        gas_concentrations: Optional[Dict[str, float]] = None,
        hotspot_temp: Optional[float] = None,
        topoil_temp: Optional[float] = None,
        load_percent: Optional[float] = None,
        rul_days: Optional[float] = None,
        failure_probability: Optional[float] = None,
        anomaly_detected: bool = False,
    ) -> List[Alert]:
        """
        Generate alerts based on transformer condition

        Args:
            transformer_id: Unique transformer identifier
            transformer_name: Transformer name
            health_score: Health index (0-100)
            fault_type: DGA fault type
            gas_concentrations: DGA gas concentrations
            hotspot_temp: Hot-spot temperature (°C)
            topoil_temp: Top-oil temperature (°C)
            load_percent: Load percentage
            rul_days: Remaining useful life (days)
            failure_probability: Failure probability (0-1)
            anomaly_detected: Whether anomaly detected

        Returns:
            List of generated alerts
        """
        alerts = []

        # DGA alerts
        if gas_concentrations:
            alerts.extend(
                self._generate_dga_alerts(
                    transformer_id, transformer_name, fault_type, gas_concentrations
                )
            )

        # Thermal alerts
        alerts.extend(
            self._generate_thermal_alerts(
                transformer_id, transformer_name, hotspot_temp, topoil_temp
            )
        )

        # Health alerts
        if health_score is not None:
            alerts.extend(
                self._generate_health_alerts(
                    transformer_id, transformer_name, health_score
                )
            )

        # Loading alerts
        if load_percent is not None:
            alerts.extend(
                self._generate_loading_alerts(
                    transformer_id, transformer_name, load_percent, hotspot_temp
                )
            )

        # Prediction alerts
        if rul_days is not None or failure_probability is not None:
            alerts.extend(
                self._generate_prediction_alerts(
                    transformer_id, transformer_name, rul_days, failure_probability
                )
            )

        # Anomaly alerts
        if anomaly_detected:
            alerts.append(self._create_anomaly_alert(transformer_id, transformer_name))

        # Sort by priority
        alerts.sort(key=lambda a: a.get_priority_order())

        # Store active alerts
        for alert in alerts:
            self.active_alerts[alert.alert_id] = alert

        return alerts

    def _generate_dga_alerts(
        self,
        transformer_id: str,
        transformer_name: str,
        fault_type: Optional[str],
        gas_concentrations: Dict[str, float],
    ) -> List[Alert]:
        """Generate DGA-related alerts"""
        alerts = []

        # Check for critical gases
        critical_gases = []
        warning_gases = []

        for gas, threshold in self.DGA_THRESHOLDS.items():
            if gas in gas_concentrations:
                conc = gas_concentrations[gas]
                if conc > threshold:
                    critical_gases.append((gas.upper(), conc, threshold))
                elif conc > threshold * 0.7:
                    warning_gases.append((gas.upper(), conc, threshold))

        # Critical DGA alert
        if critical_gases:
            if fault_type in ["D2", "T3"]:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.CRITICAL,
                    category=AlertCategory.DGA,
                    title=f"CRITICAL: {transformer_name} - {fault_type} Fault",
                    message=self._build_critical_dga_message(
                        fault_type, critical_gases
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Reduce load immediately",
                        "Schedule emergency inspection",
                        "Notify management",
                        "Prepare contingency plan",
                    ],
                    metadata={"gases": critical_gases, "fault_type": fault_type},
                )
                alerts.append(alert)
            else:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.HIGH,
                    category=AlertCategory.DGA,
                    title=f"DGA WARNING: {transformer_name}",
                    message=self._build_dga_warning_message(critical_gases),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Perform detailed DGA analysis",
                        "Schedule oil sampling",
                        "Increase monitoring frequency",
                    ],
                    metadata={"gases": critical_gases},
                )
                alerts.append(alert)

        # Warning DGA alert
        elif warning_gases:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                transformer_id=transformer_id,
                transformer_name=transformer_name,
                priority=AlertPriority.MEDIUM,
                category=AlertCategory.DGA,
                title=f"DGA MONITOR: {transformer_name}",
                message=self._build_dga_monitor_message(warning_gases),
                created_at=datetime.utcnow(),
                actions=["Continue routine DGA monitoring", "Review trend analysis"],
                metadata={"gases": warning_gases},
            )
            alerts.append(alert)

        # Fault-specific alerts
        if fault_type:
            if fault_type == "PD":
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.MEDIUM,
                    category=AlertCategory.DGA,
                    title=f"PARTIAL DISCHARGE: {transformer_name}",
                    message="Partial discharge activity detected. May indicate insulation deterioration.",
                    created_at=datetime.utcnow(),
                    actions=[
                        "Perform frequency response analysis",
                        "Inspect bushings and terminals",
                        "Review loading history",
                    ],
                    metadata={"fault_type": fault_type},
                )
                alerts.append(alert)
            elif fault_type in ["D1", "D2"]:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.HIGH,
                    category=AlertCategory.DGA,
                    title=f"DISCHARGE FAULT: {transformer_name} - {fault_type}",
                    message=self._build_discharge_message(fault_type),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Internal inspection required",
                        "Test insulation resistance",
                        "Consider oil purification",
                    ],
                    metadata={"fault_type": fault_type},
                )
                alerts.append(alert)

        return alerts

    def _build_critical_dga_message(self, fault_type: str, gases: List[tuple]) -> str:
        """Build critical DGA alert message"""
        gas_details = ", ".join(
            [f"{g[0]}: {g[1]:.1f} ppm (limit: {g[2]} ppm)" for g in gases]
        )
        return (
            f"Immediate attention required. Fault type {fault_type} detected with "
            f"critical gas concentrations: {gas_details}. "
            f"IEEE C57.104 indicates immediate action needed."
        )

    def _build_dga_warning_message(self, gases: List[tuple]) -> str:
        """Build DGA warning message"""
        gas_details = ", ".join([f"{g[0]}: {g[1]:.1f} ppm" for g in gases])
        return f"Gas concentrations exceed IEEE C57.104 limits: {gas_details}"

    def _build_dga_monitor_message(self, gases: List[tuple]) -> str:
        """Build DGA monitor message"""
        gas_details = ", ".join([f"{g[0]}: {g[1]:.1f} ppm" for g in gases])
        return (
            f"Elevated gas concentrations detected: {gas_details}. Continue monitoring."
        )

    def _build_discharge_message(self, fault_type: str) -> str:
        """Build discharge fault message"""
        if fault_type == "D2":
            return (
                "High-energy discharge (arcing) detected. Indicates severe internal fault. "
                "Immediate load reduction and inspection required."
            )
        else:
            return (
                "Low-energy discharge (sparking) detected. May indicate loose connections "
                "or insulation issues. Schedule inspection."
            )

    def _generate_thermal_alerts(
        self,
        transformer_id: str,
        transformer_name: str,
        hotspot_temp: Optional[float],
        topoil_temp: Optional[float],
    ) -> List[Alert]:
        """Generate thermal-related alerts"""
        alerts = []

        # Hot-spot temperature alerts
        if hotspot_temp is not None:
            if hotspot_temp >= self.THERMAL_THRESHOLDS["hotspot_emergency"]:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.CRITICAL,
                    category=AlertCategory.THERMAL,
                    title=f"THERMAL EMERGENCY: {transformer_name}",
                    message=(
                        f"Hot-spot temperature {hotspot_temp:.1f}°C exceeds IEEE C57.91 "
                        f"emergency limit of {self.THERMAL_THRESHOLDS['hotspot_emergency']}°C. "
                        f"Immediate load reduction required."
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        f"Reduce load to minimum immediately",
                        "Verify cooling system operation",
                        "Monitor temperature closely",
                        "Prepare for emergency shutdown if needed",
                    ],
                    metadata={"hotspot_temp": hotspot_temp},
                )
                alerts.append(alert)
            elif hotspot_temp >= self.THERMAL_THRESHOLDS["hotspot_alarm"]:
                recommended_load = max(50, 100 - (hotspot_temp - 110) * 2)
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.HIGH,
                    category=AlertCategory.THERMAL,
                    title=f"THERMAL ALARM: {transformer_name}",
                    message=(
                        f"Hot-spot temperature {hotspot_temp:.1f}°C exceeds alarm limit "
                        f"of {self.THERMAL_THRESHOLDS['hotspot_alarm']}°C. "
                        f"Recommend reducing load to {recommended_load:.0f}%."
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        f"Reduce loading to {recommended_load:.0f}%",
                        "Inspect cooling system",
                        "Check fan and pump operation",
                    ],
                    metadata={
                        "hotspot_temp": hotspot_temp,
                        "recommended_load": recommended_load,
                    },
                )
                alerts.append(alert)

        # Top-oil temperature alerts
        if topoil_temp is not None:
            if topoil_temp >= self.THERMAL_THRESHOLDS["topoil_emergency"]:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.HIGH,
                    category=AlertCategory.THERMAL,
                    title=f"TOP-OIL OVERHEATING: {transformer_name}",
                    message=(
                        f"Top-oil temperature {topoil_temp:.1f}°C exceeds emergency limit "
                        f"of {self.THERMAL_THRESHOLDS['topoil_emergency']}°C."
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Reduce load",
                        "Verify oil level",
                        "Inspect cooling equipment",
                    ],
                    metadata={"topoil_temp": topoil_temp},
                )
                alerts.append(alert)

        return alerts

    def _generate_health_alerts(
        self,
        transformer_id: str,
        transformer_name: str,
        health_score: float,
    ) -> List[Alert]:
        """Generate health index alerts"""
        alerts = []

        if health_score < self.HEALTH_THRESHOLDS["critical"]:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                transformer_id=transformer_id,
                transformer_name=transformer_name,
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.HEALTH,
                title=f"CRITICAL HEALTH: {transformer_name}",
                message=(
                    f"Health index {health_score:.1f} indicates critical condition. "
                    f"Transformer requires immediate attention and possible replacement."
                ),
                created_at=datetime.utcnow(),
                actions=[
                    "Notify management immediately",
                    "Reduce load to minimum",
                    "Plan for replacement",
                    "Review spare transformer availability",
                ],
                metadata={"health_score": health_score},
            )
            alerts.append(alert)
        elif health_score < self.HEALTH_THRESHOLDS["poor"]:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                transformer_id=transformer_id,
                transformer_name=transformer_name,
                priority=AlertPriority.HIGH,
                category=AlertCategory.HEALTH,
                title=f"POOR HEALTH: {transformer_name}",
                message=(
                    f"Health index {health_score:.1f} indicates poor condition. "
                    f"Urgent maintenance and evaluation required."
                ),
                created_at=datetime.utcnow(),
                actions=[
                    "Schedule detailed inspection",
                    "Review maintenance history",
                    "Evaluate replacement timeline",
                ],
                metadata={"health_score": health_score},
            )
            alerts.append(alert)
        elif health_score < self.HEALTH_THRESHOLDS["fair"]:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                transformer_id=transformer_id,
                transformer_name=transformer_name,
                priority=AlertPriority.MEDIUM,
                category=AlertCategory.HEALTH,
                title=f"FAIR HEALTH: {transformer_name}",
                message=(
                    f"Health index {health_score:.1f} indicates fair condition. "
                    f"Scheduled maintenance recommended."
                ),
                created_at=datetime.utcnow(),
                actions=["Schedule routine maintenance", "Continue monitoring"],
                metadata={"health_score": health_score},
            )
            alerts.append(alert)

        return alerts

    def _generate_loading_alerts(
        self,
        transformer_id: str,
        transformer_name: str,
        load_percent: float,
        hotspot_temp: Optional[float],
    ) -> List[Alert]:
        """Generate loading-related alerts"""
        alerts = []

        if load_percent > 100:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                transformer_id=transformer_id,
                transformer_name=transformer_name,
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.LOADING,
                title=f"OVERLOAD: {transformer_name}",
                message=(
                    f"Transformer loading at {load_percent:.1f}% exceeds rated capacity. "
                    f"Immediate load reduction required to prevent damage."
                ),
                created_at=datetime.utcnow(),
                actions=[
                    "Reduce load immediately below 100%",
                    "Monitor temperatures closely",
                    "Review emergency loading procedures",
                ],
                metadata={"load_percent": load_percent},
            )
            alerts.append(alert)
        elif load_percent > 90:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                transformer_id=transformer_id,
                transformer_name=transformer_name,
                priority=AlertPriority.HIGH,
                category=AlertCategory.LOADING,
                title=f"HIGH LOAD: {transformer_name}",
                message=(
                    f"Transformer loading at {load_percent:.1f}% is above recommended limit. "
                    f"Monitor temperatures closely."
                ),
                created_at=datetime.utcnow(),
                actions=[
                    "Monitor thermal conditions",
                    "Review load forecast",
                    "Prepare load shedding plan if needed",
                ],
                metadata={"load_percent": load_percent},
            )
            alerts.append(alert)

        return alerts

    def _generate_prediction_alerts(
        self,
        transformer_id: str,
        transformer_name: str,
        rul_days: Optional[float],
        failure_probability: Optional[float],
    ) -> List[Alert]:
        """Generate prediction-related alerts"""
        alerts = []

        # RUL alerts
        if rul_days is not None:
            if rul_days < 30:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.CRITICAL,
                    category=AlertCategory.PREDICTION,
                    title=f"RUL CRITICAL: {transformer_name}",
                    message=(
                        f"Estimated Remaining Useful Life: {rul_days:.0f} days. "
                        f"Immediate action required for replacement planning."
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Notify management immediately",
                        "Expedite replacement order",
                        "Prepare contingency plan",
                        "Increase inspection frequency",
                    ],
                    metadata={"rul_days": rul_days},
                )
                alerts.append(alert)
            elif rul_days < 90:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.HIGH,
                    category=AlertCategory.PREDICTION,
                    title=f"RUL WARNING: {transformer_name}",
                    message=(
                        f"Estimated Remaining Useful Life: {rul_days:.0f} days. "
                        f"Plan for maintenance or replacement."
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Begin replacement planning",
                        "Schedule detailed assessment",
                        "Budget for replacement",
                    ],
                    metadata={"rul_days": rul_days},
                )
                alerts.append(alert)

        # Failure probability alerts
        if failure_probability is not None:
            if failure_probability > 0.7:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.CRITICAL,
                    category=AlertCategory.PREDICTION,
                    title=f"HIGH FAILURE RISK: {transformer_name}",
                    message=(
                        f"Probability of failure: {failure_probability * 100:.1f}% "
                        f"within 12 months. Immediate action required."
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Implement risk mitigation measures",
                        "Increase monitoring frequency",
                        "Prepare emergency response plan",
                    ],
                    metadata={"failure_probability": failure_probability},
                )
                alerts.append(alert)
            elif failure_probability > 0.3:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    transformer_id=transformer_id,
                    transformer_name=transformer_name,
                    priority=AlertPriority.HIGH,
                    category=AlertCategory.PREDICTION,
                    title=f"FAILURE RISK ELEVATED: {transformer_name}",
                    message=(
                        f"Probability of failure: {failure_probability * 100:.1f}% "
                        f"within 12 months. Monitor closely."
                    ),
                    created_at=datetime.utcnow(),
                    actions=[
                        "Review maintenance strategy",
                        "Consider condition-based maintenance",
                    ],
                    metadata={"failure_probability": failure_probability},
                )
                alerts.append(alert)

        return alerts

    def _create_anomaly_alert(
        self,
        transformer_id: str,
        transformer_name: str,
    ) -> Alert:
        """Create anomaly detection alert"""
        return Alert(
            alert_id=str(uuid.uuid4()),
            transformer_id=transformer_id,
            transformer_name=transformer_name,
            priority=AlertPriority.MEDIUM,
            category=AlertCategory.ANOMALY,
            title=f"ANOMALY DETECTED: {transformer_name}",
            message=(
                "Unusual pattern detected in transformer operating data. "
                "Further investigation recommended."
            ),
            created_at=datetime.utcnow(),
            actions=[
                "Review recent operating history",
                "Analyze trend data",
                "Consider additional monitoring",
            ],
        )

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "System") -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Who is acknowledging the alert

        Returns:
            True if acknowledged successfully
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledge(acknowledged_by)
            return True
        return False

    def get_active_alerts(
        self,
        transformer_id: Optional[str] = None,
        priority: Optional[AlertPriority] = None,
        unacknowledged_only: bool = False,
    ) -> List[Alert]:
        """
        Get active alerts with optional filters

        Args:
            transformer_id: Filter by transformer
            priority: Filter by priority
            unacknowledged_only: Only return unacknowledged alerts

        Returns:
            List of alerts
        """
        alerts = list(self.active_alerts.values())

        if transformer_id:
            alerts = [a for a in alerts if a.transformer_id == transformer_id]

        if priority:
            alerts = [a for a in alerts if a.priority == priority]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        # Sort by priority
        alerts.sort(key=lambda a: a.get_priority_order())

        return alerts

    def get_alert_summary(self, transformer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of active alerts

        Args:
            transformer_id: Optional filter by transformer

        Returns:
            Summary dictionary
        """
        alerts = self.get_active_alerts(transformer_id=transformer_id)

        summary = {
            "total": len(alerts),
            "unacknowledged": len([a for a in alerts if not a.acknowledged]),
            "by_priority": {
                "CRITICAL": len(
                    [a for a in alerts if a.priority == AlertPriority.CRITICAL]
                ),
                "HIGH": len([a for a in alerts if a.priority == AlertPriority.HIGH]),
                "MEDIUM": len(
                    [a for a in alerts if a.priority == AlertPriority.MEDIUM]
                ),
                "LOW": len([a for a in alerts if a.priority == AlertPriority.LOW]),
                "INFO": len([a for a in alerts if a.priority == AlertPriority.INFO]),
            },
            "by_category": {
                "DGA": len([a for a in alerts if a.category == AlertCategory.DGA]),
                "THERMAL": len(
                    [a for a in alerts if a.category == AlertCategory.THERMAL]
                ),
                "HEALTH": len(
                    [a for a in alerts if a.category == AlertCategory.HEALTH]
                ),
                "LOADING": len(
                    [a for a in alerts if a.category == AlertCategory.LOADING]
                ),
                "ANOMALY": len(
                    [a for a in alerts if a.category == AlertCategory.ANOMALY]
                ),
                "PREDICTION": len(
                    [a for a in alerts if a.category == AlertCategory.PREDICTION]
                ),
                "MAINTENANCE": len(
                    [a for a in alerts if a.category == AlertCategory.MAINTENANCE]
                ),
            },
            "alerts": [a.to_dict() for a in alerts],
        }

        return summary
