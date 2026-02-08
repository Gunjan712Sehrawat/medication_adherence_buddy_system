"""
Ensemble Risk Scoring Model
Combines rule-based and ML approaches for optimal performance
"""
print("✅ USING ensemble.py FROM:", __file__)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from typing import Dict, Optional
import shap

from .rule_engine import RuleEngine


class EnsembleRiskScorer:
    """
    Hybrid ensemble combining:
    1. Rule-based scoring
    2. LightGBM model
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)

        self.rule_engine = RuleEngine()
        self.ml_model = None
        self.scaler = StandardScaler()
        self.explainer = None

        self.ensemble_weights = {
            "rules": 0.4,
            "ml": 0.6
        }

        self.feature_importance = None

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("⚠️ Config not found, using defaults")
            return {
                "model": {
                    "hyperparameters": {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "learning_rate": 0.05,
                        "min_child_samples": 20,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8
                    }
                }
            }

    # FEATURE EXTRACTION
    def _extract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.empty:
            return X.copy()
        
        features = X.copy()

        # Drop identifiers
        for col in ["patient_id"]:
            if col in features.columns:
                features.drop(columns=[col], inplace=True)

        # Convert list-type columns → numeric
        for col in features.columns:
            if features[col].apply(lambda x: isinstance(x, list)).any():
                features[col] = features[col].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )

        # Derived features
        if "ivr_calls_total" in features.columns:
            features["ivr_response_rate"] = (
                features.get("ivr_calls_answered", 0) /
                features["ivr_calls_total"].replace(0, 1)
            )
            features["ivr_miss_rate"] = (
                features.get("ivr_calls_missed", 0) /
                features["ivr_calls_total"].replace(0, 1)
            )

        if "sms_delivered" in features.columns:
            features["sms_read_rate"] = (
                features.get("sms_read", 0) /
                features["sms_delivered"].replace(0, 1)
            )

        if "days_since_prescription" in features.columns:
            features["early_treatment"] = (features["days_since_prescription"] < 14).astype(int)
            features["established_treatment"] = (features["days_since_prescription"] > 90).astype(int)

        if {"ivr_miss_rate", "sms_read_rate"}.issubset(features.columns):
            features["overall_engagement"] = (
                (1 - features["ivr_miss_rate"]) * 0.5 +
                features["sms_read_rate"] * 0.5
            )

        # Fill NA ONLY for numeric columns
        numeric_cols = features.select_dtypes(include=["number"]).columns
        
        if len(numeric_cols) > 0:
            medians = features[numeric_cols].median()
            features[numeric_cols] = features[numeric_cols].fillna(medians)

        return features

    # TRAIN
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        print("Training Ensemble Risk Scorer...")

        X_train_feats = self._extract_features(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_feats)

        params = self.config["model"]["hyperparameters"]

        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": params["learning_rate"],
            "num_leaves": 2 ** params["max_depth"],
            "feature_fraction": params["colsample_bytree"],
            "bagging_fraction": params["subsample"],
            "min_child_samples": params["min_child_samples"],
            "verbose": -1
        }

        train_data = lgb.Dataset(X_train_scaled, label=y_train)

        if X_val is not None and y_val is not None:
            X_val_feats = self._extract_features(X_val)
            X_val_scaled = self.scaler.transform(X_val_feats)
            val_data = lgb.Dataset(X_val_scaled, label=y_val)

            self.ml_model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=params["n_estimators"],
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(20)]
            )
        else:
            self.ml_model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=params["n_estimators"]
            )

        self.explainer = shap.TreeExplainer(self.ml_model)

        self.feature_importance = pd.DataFrame({
            "feature": X_train_feats.columns,
            "importance": self.ml_model.feature_importance(importance_type="gain")
        }).sort_values("importance", ascending=False)

        print("✓ Training complete")

    # PREDICT
    def predict(self, X: pd.DataFrame, return_components: bool = False):
        rule_scores = np.array([
            self.rule_engine.predict(row.to_dict())[0]
            for _, row in X.iterrows()
        ])

        X_feats = self._extract_features(X)
        X_scaled = self.scaler.transform(X_feats)
        ml_scores = self.ml_model.predict(X_scaled)

        ensemble = (
            self.ensemble_weights["rules"] * rule_scores +
            self.ensemble_weights["ml"] * ml_scores
        )

        if return_components:
            return {
                "ensemble": ensemble,
                "rules": rule_scores,
                "ml": ml_scores
            }

        return ensemble

    # EXPLAIN
    def explain(self, X: pd.DataFrame, patient_idx: int = 0) -> Dict:
        """
        Unified explanation combining rule engine + ML
        """
        row = X.iloc[patient_idx]

        # Rule-based explanation
        rule_score, rule_factors = self.rule_engine.predict(row.to_dict())

        # ML score
        X_feats = self._extract_features(X.iloc[[patient_idx]])
        X_scaled = self.scaler.transform(X_feats)
        ml_score = float(self.ml_model.predict(X_scaled)[0])

        # Ensemble score
        ensemble_score = (
            self.ensemble_weights["rules"] * rule_score +
            self.ensemble_weights["ml"] * ml_score
        )

        # Risk level
        if ensemble_score >= 0.7:
            level = "HIGH"
        elif ensemble_score >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "risk_score": ensemble_score,
            "risk_level": level,
            "component_scores": {
                "rule_based": rule_score,
                "ml_model": ml_score
            },
            "rule_explanation": {
                "triggered_factors": rule_factors,
                "recommendations": self.rule_engine._get_recommendations(
                    ensemble_score, rule_factors
                )
            }
        }


    # SAVE 
    def save(self, filepath: str):
        """Save trained model safely (no lambdas pickled)"""
        model_data = {
            "ml_model": self.ml_model,
            "scaler": self.scaler,
            "ensemble_weights": self.ensemble_weights,
            "feature_importance": self.feature_importance,
            "config": self.config
        }

        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    # LOAD
    def load(self, filepath: str):
        """Load trained model safely"""
        model_data = joblib.load(filepath)

        self.ml_model = model_data["ml_model"]
        self.scaler = model_data["scaler"]
        self.ensemble_weights = model_data["ensemble_weights"]
        self.feature_importance = model_data["feature_importance"]
        self.config = model_data["config"]

        # Recreate rule engine (DO NOT LOAD IT)
        self.rule_engine = RuleEngine()

        # Recreate SHAP explainer
        if self.ml_model is not None:
            self.explainer = shap.TreeExplainer(self.ml_model)

        print(f"✓ Model loaded from {filepath}")

    
    # GET FEATURE IMPORTANCE

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
            """
            Return top-N feature importance from ML model
            Used by train_model.py for visualization
            """
            if self.feature_importance is None:
                return pd.DataFrame(columns=["feature", "importance"])

            return self.feature_importance.head(top_n)




