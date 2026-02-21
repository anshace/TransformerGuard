"""
ML-Based DGA Fault Classifier.

Uses machine learning (Random Forest) to classify transformer faults
based on DGA gas concentrations. Can be trained on historical data
with known fault types.

Reference: IEEE C57.104-2019

Author: TransformerGuard Team
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .duval_triangle import FaultType

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MLClassifierResult:
    """
    Result from ML fault classification.

    Attributes:
        fault_type: Predicted fault type
        confidence: Confidence score (0.0 to 1.0)
        probabilities: Probability for each fault type
        method: Name of the ML method used
        explanation: Human-readable explanation of the prediction
    """

    fault_type: FaultType
    confidence: float
    probabilities: Dict[str, float]
    method: str
    explanation: str


class DGAFaultClassifier:
    """
    Machine Learning classifier for DGA fault diagnosis.

    Uses a Random Forest classifier trained on historical DGA data
    with known fault types. Provides probability estimates for each
    fault category.

    The classifier uses 7 key gases as features:
    - H2 (Hydrogen)
    - CH4 (Methane)
    - C2H6 (Ethane)
    - C2H4 (Ethylene)
    - C2H2 (Acetylene)
    - CO (Carbon Monoxide)
    - CO2 (Carbon Dioxide)

    Example:
        >>> classifier = DGAFaultClassifier()
        >>> classifier = create_default_classifier()  # Use pre-trained
        >>> result = classifier.predict(h2=150, ch4=200, c2h2=10,
        ...                             c2h4=50, c2h6=30, co=250, co2=1500)
        >>> print(result.fault_type)
        FaultType.D2
    """

    # Feature names for the model
    FEATURES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]

    # Fault type mapping (label index to FaultType enum)
    FAULT_MAPPING = {
        0: FaultType.NORMAL,
        1: FaultType.PD,
        2: FaultType.D1,
        3: FaultType.D2,
        4: FaultType.T1,
        5: FaultType.T2,
        6: FaultType.T3,
        7: FaultType.DT,
    }

    # Reverse mapping (FaultType to label index)
    FAULT_TO_INDEX = {v: k for k, v in FAULT_MAPPING.items()}

    # Default hyperparameters for Random Forest
    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to a pre-trained model file. If provided,
                       the model will be loaded from this path.
        """
        self._model = None
        self._is_trained = False
        self._params = self.DEFAULT_PARAMS.copy()

        if model_path is not None:
            self.load_model(model_path)

    @property
    def is_trained(self) -> bool:
        """Check if the model is trained and ready for predictions."""
        return self._is_trained

    @property
    def is_available(self) -> bool:
        """Check if sklearn is available."""
        try:
            from sklearn.ensemble import RandomForestClassifier  # noqa: F401

            return True
        except ImportError:
            logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
            return False

    def _create_model(self):
        """Create the underlying RandomForest model."""
        if self.is_available:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=self._params["n_estimators"],
                max_depth=self._params["max_depth"],
                min_samples_split=self._params["min_samples_split"],
                min_samples_leaf=self._params["min_samples_leaf"],
                class_weight=self._params.get("class_weight"),
                random_state=self._params.get("random_state", 42),
            )
        return None

    def train(
        self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the classifier on historical DGA data.

        Args:
            X: Feature matrix (n_samples, n_features)
               Columns: [h2, ch4, c2h6, c2h4, c2h2, co, co2]
            y: Labels (0-7 corresponding to fault types, or FaultType enums)
            save_path: Optional path to save the trained model

        Returns:
            Training metrics dictionary containing:
            - accuracy: Training accuracy
            - n_samples: Number of training samples
            - n_features: Number of features
            - feature_importance: Dictionary of feature importance scores

        Raises:
            ValueError: If X or y have invalid shapes
            RuntimeError: If sklearn is not available
        """
        if not self.is_available:
            raise RuntimeError(
                "scikit-learn is required for training. "
                "Install with: pip install scikit-learn"
            )

        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if X.shape[1] != len(self.FEATURES):
            raise ValueError(
                f"X must have {len(self.FEATURES)} features (columns), "
                f"got {X.shape[1]}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {len(X)}, y: {len(y)}"
            )

        # Convert FaultType enums to indices if needed
        y_numeric = np.zeros(len(y), dtype=int)
        for i, label in enumerate(y):
            if isinstance(label, FaultType):
                y_numeric[i] = self.FAULT_TO_INDEX.get(label, 0)
            else:
                y_numeric[i] = int(label)

        logger.info(f"Training classifier with {len(X)} samples...")

        # Create and train model
        self._model = self._create_model()
        self._model.fit(X, y_numeric)
        self._is_trained = True

        # Calculate training accuracy
        y_pred = self._model.predict(X)
        accuracy = np.mean(y_pred == y_numeric)

        # Get feature importance
        feature_importance = self.get_feature_importance()

        logger.info(f"Training complete. Accuracy: {accuracy:.2%}")

        # Save model if path provided
        if save_path is not None:
            self.save_model(save_path)

        return {
            "accuracy": accuracy,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_importance": feature_importance,
        }

    def predict(
        self,
        h2: float,
        ch4: float,
        c2h6: float,
        c2h4: float,
        c2h2: float,
        co: float,
        co2: float,
    ) -> MLClassifierResult:
        """
        Classify fault type from DGA gas concentrations.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h6: Ethane concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h2: Acetylene concentration in ppm
            co: Carbon monoxide concentration in ppm
            co2: Carbon dioxide concentration in ppm

        Returns:
            MLClassifierResult with predicted fault type and confidence

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model is not trained. Call train() first or load a pre-trained model."
            )

        # Create feature array
        X = np.array([[h2, ch4, c2h6, c2h4, c2h2, co, co2]])

        # Get prediction and probabilities
        pred_index = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]

        # Map to fault type
        fault_type = self.FAULT_MAPPING.get(pred_index, FaultType.UNDETERMINED)
        confidence = float(proba[pred_index])

        # Build probability dictionary
        probabilities = {}
        for idx, prob in enumerate(proba):
            fault_name = self.FAULT_MAPPING.get(idx, FaultType.UNDETERMINED)
            probabilities[fault_name.name] = float(prob)

        # Generate explanation
        explanation = self._generate_explanation(
            fault_type, confidence, probabilities, h2, ch4, c2h6, c2h4, c2h2, co, co2
        )

        return MLClassifierResult(
            fault_type=fault_type,
            confidence=confidence,
            probabilities=probabilities,
            method="random_forest",
            explanation=explanation,
        )

    def predict_batch(self, gas_data: np.ndarray) -> List[MLClassifierResult]:
        """
        Classify multiple samples at once.

        Args:
            gas_data: Array of shape (n_samples, 7) with gas concentrations
                     Columns: [h2, ch4, c2h6, c2h4, c2h2, co, co2]

        Returns:
            List of MLClassifierResult for each sample

        Raises:
            RuntimeError: If model is not trained
            ValueError: If gas_data has invalid shape
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model is not trained. Call train() first or load a pre-trained model."
            )

        if gas_data.ndim != 2 or gas_data.shape[1] != len(self.FEATURES):
            raise ValueError(
                f"gas_data must have shape (n_samples, {len(self.FEATURES)}), "
                f"got {gas_data.shape}"
            )

        # Get predictions and probabilities
        pred_indices = self._model.predict(gas_data)
        probas = self._model.predict_proba(gas_data)

        results = []
        for i in range(len(gas_data)):
            pred_index = pred_indices[i]
            proba = probas[i]

            fault_type = self.FAULT_MAPPING.get(pred_index, FaultType.UNDETERMINED)
            confidence = float(proba[pred_index])

            # Build probability dictionary
            probabilities = {}
            for idx, prob in enumerate(proba):
                fault_name = self.FAULT_MAPPING.get(idx, FaultType.UNDETERMINED)
                probabilities[fault_name.name] = float(prob)

            # Get gas values for explanation
            h2, ch4, c2h6, c2h4, c2h2, co, co2 = gas_data[i]

            explanation = self._generate_explanation(
                fault_type, confidence, probabilities, h2, ch4, c2h6, c2h4, c2h2, co, co2
            )

            results.append(
                MLClassifierResult(
                    fault_type=fault_type,
                    confidence=confidence,
                    probabilities=probabilities,
                    method="random_forest",
                    explanation=explanation,
                )
            )

        return results

    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model from file.

        Args:
            model_path: Path to the saved model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model file is invalid or incompatible
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, "rb") as f:
                saved_data = pickle.load(f)

            if isinstance(saved_data, dict):
                # New format with metadata
                self._model = saved_data.get("model")
                self._params = saved_data.get("params", self.DEFAULT_PARAMS.copy())
            else:
                # Legacy format (just the model)
                self._model = saved_data

            self._is_trained = True
            logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def save_model(self, model_path: str) -> None:
        """
        Save the current model to file.

        Args:
            model_path: Path to save the model file

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model.")

        # Create directory if needed
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)

        # Save model with metadata
        saved_data = {
            "model": self._model,
            "params": self._params,
            "features": self.FEATURES,
            "fault_mapping": self.FAULT_MAPPING,
        }

        with open(model_path, "wb") as f:
            pickle.dump(saved_data, f)

        logger.info(f"Model saved to {model_path}")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained.")

        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            return {name: float(imp) for name, imp in zip(self.FEATURES, importances)}

        return {name: 0.0 for name in self.FEATURES}

    def _generate_explanation(
        self,
        fault_type: FaultType,
        confidence: float,
        probabilities: Dict[str, float],
        h2: float,
        ch4: float,
        c2h6: float,
        c2h4: float,
        c2h2: float,
        co: float,
        co2: float,
    ) -> str:
        """Generate human-readable explanation of the prediction."""
        # Get top 3 probabilities
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]

        explanation = f"=== ML-Based DGA Fault Classification ===\n\n"
        explanation += f"Primary Diagnosis: {fault_type.value}\n"
        explanation += f"Confidence: {confidence:.1%}\n\n"

        explanation += "Input Gas Concentrations (ppm):\n"
        explanation += f"  H2: {h2:.1f}, CH4: {ch4:.1f}, C2H6: {c2h6:.1f}\n"
        explanation += f"  C2H4: {c2h4:.1f}, C2H2: {c2h2:.1f}\n"
        explanation += f"  CO: {co:.1f}, CO2: {co2:.1f}\n\n"

        explanation += "Top Predicted Fault Types:\n"
        for fault_name, prob in sorted_probs:
            explanation += f"  {fault_name}: {prob:.1%}\n"

        # Add fault-specific interpretation
        explanation += "\n--- Interpretation ---\n"

        if fault_type == FaultType.NORMAL:
            explanation += "Gas concentrations are within normal operating range. "
            explanation += "No significant fault indicators detected."
        elif fault_type == FaultType.PD:
            explanation += "Partial Discharge indicated. High H2 with low C2H2 suggests "
            explanation += "low-energy electrical discharge, possibly due to voids or "
            explanation += "delamination in insulation."
        elif fault_type == FaultType.D1:
            explanation += "Low Energy Discharge (D1) indicated. Presence of C2H2 with "
            explanation += "moderate C2H4 suggests sporadic arcing or spark discharge, "
            explanation += "possibly from floating potentials."
        elif fault_type == FaultType.D2:
            explanation += "High Energy Discharge (D2) indicated. High C2H2 with "
            explanation += "significant C2H4 suggests high-energy arcing. This is a "
            explanation += "severe condition requiring immediate attention."
        elif fault_type == FaultType.T1:
            explanation += "Low Temperature Thermal Fault (<300°C) indicated. "
            explanation += "Predominance of CH4 with low C2H4 suggests minor overheating, "
            explanation += "possibly from loose connections or circulating currents."
        elif fault_type == FaultType.T2:
            explanation += "Medium Temperature Thermal Fault (300-700°C) indicated. "
            explanation += "Significant C2H4 production suggests overheating in the "
            explanation += "300-700°C range. Investigate core and winding connections."
        elif fault_type == FaultType.T3:
            explanation += "High Temperature Thermal Fault (>700°C) indicated. "
            explanation += "High C2H4 with some C2H2 suggests severe overheating. "
            explanation += "This condition requires urgent investigation."
        elif fault_type == FaultType.DT:
            explanation += "Mixed Discharge/Thermal fault indicated. Combination of "
            explanation += "C2H2 and C2H4 suggests both electrical and thermal faults "
            explanation += "are present simultaneously."
        else:
            explanation += "Unable to determine fault type with confidence. "
            explanation += "Consider additional diagnostic methods."

        return explanation


def create_default_classifier() -> DGAFaultClassifier:
    """
    Create a classifier with default pre-trained weights.

    This uses synthetic training data based on IEEE C57.104
    typical gas profiles for each fault type. The synthetic data
    is generated from known gas concentration patterns for each
    fault category.

    Returns:
        DGAFaultClassifier: Pre-trained classifier ready for predictions

    Note:
        The default classifier is trained on synthetic data and should
        be replaced with a model trained on actual historical data when
        available. Synthetic data is based on typical gas profiles from
        IEEE C57.104-2019.
    """
    logger.info("Creating default classifier with synthetic training data...")

    # IEEE C57.104 typical gas profiles for each fault type
    # These are approximate values based on the standard
    # Format: [H2, CH4, C2H6, C2H4, C2H2, CO, CO2]

    # Normal operation - low gas concentrations
    normal_profiles = [
        [20, 30, 10, 5, 1, 100, 500],
        [25, 35, 12, 6, 0.5, 120, 600],
        [15, 25, 8, 4, 0.3, 80, 400],
        [30, 40, 15, 8, 1, 150, 700],
        [18, 28, 9, 5, 0.8, 90, 450],
    ]

    # Partial Discharge - high H2, low C2H2
    pd_profiles = [
        [500, 100, 30, 20, 5, 150, 800],
        [600, 120, 35, 25, 3, 180, 900],
        [450, 90, 25, 15, 4, 130, 700],
        [700, 150, 40, 30, 6, 200, 1000],
        [550, 110, 32, 22, 4, 160, 850],
    ]

    # Low Energy Discharge (D1) - moderate C2H2, low C2H4
    d1_profiles = [
        [150, 80, 25, 30, 50, 120, 600],
        [180, 90, 30, 35, 60, 140, 700],
        [130, 70, 20, 25, 45, 100, 550],
        [200, 100, 35, 40, 70, 160, 800],
        [160, 85, 28, 32, 55, 130, 650],
    ]

    # High Energy Discharge (D2) - high C2H2, high C2H4
    d2_profiles = [
        [300, 150, 50, 200, 150, 200, 1000],
        [350, 180, 60, 250, 180, 250, 1200],
        [280, 130, 45, 180, 130, 180, 900],
        [400, 200, 70, 300, 200, 300, 1500],
        [320, 160, 55, 220, 160, 220, 1100],
    ]

    # Thermal Fault T1 (<300°C) - high CH4, low C2H4
    t1_profiles = [
        [100, 300, 100, 30, 3, 200, 1000],
        [120, 350, 120, 35, 4, 250, 1200],
        [90, 280, 90, 25, 2, 180, 900],
        [140, 400, 140, 40, 5, 300, 1500],
        [110, 320, 110, 32, 3, 220, 1100],
    ]

    # Thermal Fault T2 (300-700°C) - moderate C2H4
    t2_profiles = [
        [150, 250, 80, 150, 5, 250, 1200],
        [180, 300, 100, 180, 6, 300, 1500],
        [130, 220, 70, 130, 4, 200, 1000],
        [200, 350, 120, 200, 7, 350, 1800],
        [160, 270, 90, 160, 5, 270, 1300],
    ]

    # Thermal Fault T3 (>700°C) - high C2H4
    t3_profiles = [
        [200, 300, 100, 500, 10, 300, 1500],
        [250, 350, 120, 600, 12, 350, 1800],
        [180, 280, 90, 450, 8, 250, 1200],
        [300, 400, 140, 700, 15, 400, 2000],
        [220, 320, 110, 550, 11, 320, 1600],
    ]

    # Mixed Discharge/Thermal (DT) - combination
    dt_profiles = [
        [250, 200, 60, 200, 80, 250, 1200],
        [300, 250, 70, 250, 100, 300, 1500],
        [220, 180, 50, 180, 70, 200, 1000],
        [350, 280, 80, 300, 120, 350, 1800],
        [270, 220, 65, 220, 90, 270, 1300],
    ]

    # Combine all profiles
    all_profiles = (
        normal_profiles
        + pd_profiles
        + d1_profiles
        + d2_profiles
        + t1_profiles
        + t2_profiles
        + t3_profiles
        + dt_profiles
    )

    # Create labels (0-7 for each fault type)
    labels = (
        [0] * len(normal_profiles)
        + [1] * len(pd_profiles)
        + [2] * len(d1_profiles)
        + [3] * len(d2_profiles)
        + [4] * len(t1_profiles)
        + [5] * len(t2_profiles)
        + [6] * len(t3_profiles)
        + [7] * len(dt_profiles)
    )

    # Add noise and variations to create more training samples
    np.random.seed(42)
    X_augmented = []
    y_augmented = []

    for profile, label in zip(all_profiles, labels):
        # Original sample
        X_augmented.append(profile)
        y_augmented.append(label)

        # Add 10 noisy variations
        for _ in range(10):
            noise = np.random.normal(0, 0.1, 7)  # 10% noise
            noisy_profile = np.array(profile) * (1 + noise)
            noisy_profile = np.maximum(noisy_profile, 0)  # Ensure non-negative
            X_augmented.append(noisy_profile.tolist())
            y_augmented.append(label)

    X = np.array(X_augmented)
    y = np.array(y_augmented)

    # Create and train classifier
    classifier = DGAFaultClassifier()
    classifier.train(X, y)

    logger.info(f"Default classifier created with {len(X)} synthetic training samples")

    return classifier