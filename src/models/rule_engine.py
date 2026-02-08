"""
Rule-Based Risk Scoring Engine
Provides interpretable, clinically-grounded risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

# RULE CONDITIONS 

def high_ivr_miss_rate(x: Dict) -> bool:
    missed = x.get("ivr_calls_missed", 0)
    total = max(x.get("ivr_calls_total", 1), 1)
    return (missed / total) > 0.6


def no_ivr_response(x: Dict) -> bool:
    return x.get("ivr_calls_answered", 0) == 0 and x.get("ivr_calls_total", 0) >= 3


def declining_ivr_response(x: Dict) -> bool:
    pattern = x.get("ivr_response_pattern", [])
    if len(pattern) < 3:
        return False
    recent = pattern[-3:]
    earlier = pattern[:3] if len(pattern) >= 6 else pattern[:-3]
    return np.mean(recent) < np.mean(earlier) * 0.7


def low_sms_read_rate(x: Dict) -> bool:
    read = x.get("sms_read", 0)
    delivered = max(x.get("sms_delivered", 1), 1)
    return (read / delivered) < 0.4


def slow_sms_response(x: Dict) -> bool:
    return x.get("sms_avg_response_time_hrs", 0) > 24


def no_sms_engagement(x: Dict) -> bool:
    return x.get("sms_delivered", 0) >= 5 and x.get("sms_read", 0) == 0


def missed_refill_window(x: Dict) -> bool:
    return x.get("days_since_refill_due", 0) > 3


def early_treatment_phase(x: Dict) -> bool:
    return x.get("days_since_prescription", 0) < 14


def weekend_nonadherence(x: Dict) -> bool:
    return x.get("weekend_response_rate", 1.0) < 0.5


def high_risk_age(x: Dict) -> bool:
    age = x.get("age", 50)
    return age > 75 or age < 25


def multiple_medications(x: Dict) -> bool:
    return x.get("num_medications", 0) >= 5


def complex_regimen(x: Dict) -> bool:
    return x.get("doses_per_day", 1) >= 3


# DATA STRUCTURE

@dataclass
class RiskRule:
    name: str
    condition: callable
    score: float
    explanation: str
    category: str


# RULE ENGINE

class RuleEngine:
    """
    Rule-based risk scoring system for medication non-adherence.
    """

    def __init__(self):
        self.rules = self._initialize_rules()
        self.weights = {
            "ivr": 0.35,
            "sms": 0.30,
            "behavioral": 0.25,
            "demographic": 0.10
        }

    def _initialize_rules(self) -> List[RiskRule]:
        return [
            # IVR
            RiskRule("high_ivr_miss_rate", high_ivr_miss_rate, 0.8,
                     "Missed >60% of IVR reminder calls", "ivr"),
            RiskRule("no_ivr_response", no_ivr_response, 0.9,
                     "No response to 3+ IVR calls", "ivr"),
            RiskRule("declining_ivr_response", declining_ivr_response, 0.7,
                     "Declining IVR response pattern over time", "ivr"),

            # SMS
            RiskRule("low_sms_read_rate", low_sms_read_rate, 0.6,
                     "Read <40% of SMS reminders", "sms"),
            RiskRule("slow_sms_response", slow_sms_response, 0.5,
                     "Average SMS response time >24 hours", "sms"),
            RiskRule("no_sms_engagement", no_sms_engagement, 0.85,
                     "No engagement with 5+ SMS messages", "sms"),

            # Behavioral
            RiskRule("missed_refill_window", missed_refill_window, 0.9,
                     "Prescription refill overdue by >3 days", "behavioral"),
            RiskRule("early_treatment_phase", early_treatment_phase, 0.4,
                     "First 2 weeks - higher discontinuation risk", "behavioral"),
            RiskRule("weekend_nonadherence", weekend_nonadherence, 0.5,
                     "Low medication adherence on weekends", "behavioral"),

            # Demographic
            RiskRule("high_risk_age", high_risk_age, 0.3,
                     "Age group with higher non-adherence rates", "demographic"),
            RiskRule("multiple_medications", multiple_medications, 0.4,
                     "Polypharmacy (5+ medications) increases risk", "demographic"),
            RiskRule("complex_regimen", complex_regimen, 0.35,
                     "Complex dosing schedule (3+ times daily)", "demographic"),
        ]

    # PREDICTION

    def predict(self, patient_data: Dict) -> Tuple[float, List[str]]:
        triggered = []
        category_scores = {k: [] for k in self.weights}

        for rule in self.rules:
            try:
                if rule.condition(patient_data):
                    triggered.append(rule)
                    category_scores[rule.category].append(rule.score)
            except Exception:
                continue

        risk_score = 0.0
        for cat, weight in self.weights.items():
            if category_scores[cat]:
                risk_score += max(category_scores[cat]) * weight

        risk_score = min(risk_score, 1.0)

        explanations = [r.explanation for r in sorted(
            triggered, key=lambda r: r.score, reverse=True
        )]

        return risk_score, explanations

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for idx, row in df.iterrows():
            score, explanations = self.predict(row.to_dict())
            results.append({
                "patient_id": row.get("patient_id", idx),
                "risk_score": score,
                "risk_level": self._categorize_risk(score),
                "top_factors": explanations[:3],
                "num_triggered_rules": len(explanations)
            })
        return pd.DataFrame(results)

    def _categorize_risk(self, score: float) -> str:
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        return "LOW"
