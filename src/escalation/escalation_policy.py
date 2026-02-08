"""
Escalation Policy Engine
Separates AI prediction from human decision-making and intervention

Key Principle: AI PREDICTS risk, HUMANS DECIDE on interventions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels for human review queue"""
    URGENT = "urgent"           # Review within 4 hours
    HIGH = "high"               # Review within 24 hours
    MEDIUM = "medium"           # Review within 48 hours
    LOW = "low"                 # Review within 1 week
    MONITOR = "monitor"         # Passive monitoring only


class InterventionType(Enum):
    """Types of human interventions available"""
    PHONE_CALL = "phone_call"
    SMS_FOLLOWUP = "sms_followup"
    PHARMACIST_CONSULT = "pharmacist_consult"
    PROVIDER_NOTIFICATION = "provider_notification"
    HOME_VISIT = "home_visit"
    NO_ACTION = "no_action"


class EscalationPolicy:
    """
    Manages the escalation from AI risk prediction to human decision-making.
    
    CRITICAL SEPARATION:
    - AI System: Predicts risk scores (0-1)
    - This Module: Translates scores to review priorities
    - Human Clinicians: Make final decisions on interventions
    
    The AI NEVER directly triggers patient contact. It only creates
    review tasks for qualified healthcare staff.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize escalation policy
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or self._default_config()
        
        # Track review queue for audit
        self.review_queue = []
        
        # Track intervention decisions for outcome analysis
        self.intervention_log = []
    
    def _default_config(self) -> Dict:
        """Default escalation policy configuration"""
        return {
            # Risk score thresholds for different priorities
            'thresholds': {
                'urgent': 0.85,      # Extreme risk - immediate review
                'high': 0.70,        # High risk - review today
                'medium': 0.50,      # Moderate risk - review this week
                'low': 0.30,         # Mild concern - monitor
                'monitor': 0.0       # Low risk - passive monitoring
            },
            
            # Maximum alerts per priority level (prevent overwhelming staff)
            'daily_limits': {
                'urgent': 10,
                'high': 25,
                'medium': 50,
                'low': 100
            },
            
            # Required review times
            'review_deadlines': {
                'urgent': 4,         # hours
                'high': 24,          # hours
                'medium': 48,        # hours
                'low': 168,          # hours (1 week)
            },
            
            # Escalation modifiers (boost priority based on context)
            'priority_boosters': {
                'recent_hospitalization': 1.15,
                'multiple_chronic_conditions': 1.10,
                'high_risk_medication': 1.20,
                'elderly_living_alone': 1.12,
                'prior_adherence_issues': 1.08,
                'no_recent_contact': 1.10
            },
            
            # De-escalation criteria (reduce alert fatigue)
            'deescalation_rules': {
                'recent_successful_contact': -0.1,
                'improving_trend': -0.15,
                'low_complexity_regimen': -0.05
            }
        }
    
    def create_review_task(self, patient_id: str, risk_score: float, 
                          risk_factors: List[str], 
                          patient_context: Optional[Dict] = None) -> Dict:
        """
        Create a review task for human clinicians.
        
        THIS IS THE KEY SEPARATION POINT:
        - Input: AI risk prediction
        - Output: Human review task (NOT automated intervention)
        
        Args:
            patient_id: Patient identifier
            risk_score: AI-predicted risk score (0-1)
            risk_factors: Explanation of risk factors
            patient_context: Additional patient information
            
        Returns:
            Review task dictionary for clinical review queue
        """
        
        # Apply contextual adjustments
        adjusted_score = self._apply_contextual_adjustments(
            risk_score, patient_context or {}
        )
        
        # Determine priority level
        priority = self._determine_priority(adjusted_score)
        
        # Check daily limits (prevent alert fatigue)
        if self._exceeds_daily_limit(priority):
            logger.warning(
                f"Daily limit reached for {priority.value} priority. "
                f"Downgrading to lower priority."
            )
            priority = self._downgrade_priority(priority)
        
        # Calculate review deadline
        deadline = self._calculate_deadline(priority)
        
        # Create review task
        review_task = {
            'task_id': f"REVIEW_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'patient_id': patient_id,
            'created_at': datetime.now().isoformat(),
            'review_deadline': deadline.isoformat(),
            
            # AI Prediction (for context only)
            'ai_prediction': {
                'risk_score': round(risk_score, 3),
                'adjusted_score': round(adjusted_score, 3),
                'risk_factors': risk_factors,
                'model_version': '1.0.0'
            },
            
            # Priority for human review
            'priority': priority.value,
            'priority_explanation': self._explain_priority(
                adjusted_score, risk_factors, patient_context
            ),
            
            # Review assignment
            'status': 'pending_review',
            'assigned_to': None,  # Will be assigned by workflow manager
            'reviewed_by': None,
            'reviewed_at': None,
            
            # Human decision (to be filled by clinician)
            'clinical_decision': {
                'decision_made': False,
                'intervention_type': None,
                'intervention_scheduled': None,
                'decision_rationale': None,
                'override_ai': False,
                'override_reason': None
            },
            
            # Audit trail
            'metadata': {
                'patient_context': patient_context,
                'queue_position': None,
                'escalation_policy_version': '1.0.0'
            }
        }
        
        # Add to review queue
        self.review_queue.append(review_task)
        
        logger.info(
            f"Review task created: {review_task['task_id']} | "
            f"Priority: {priority.value} | "
            f"Deadline: {deadline.strftime('%Y-%m-%d %H:%M')}"
        )
        
        return review_task
    
    def _apply_contextual_adjustments(self, risk_score: float, 
                                     context: Dict) -> float:
        """
        Apply contextual adjustments to risk score.
        
        This adjusts priority based on patient-specific factors
        that the base AI model may not fully capture.
        """
        adjusted = risk_score
        boosters = self.config['priority_boosters']
        deescalators = self.config['deescalation_rules']
        
        # Apply boosters (increase priority)
        if context.get('recent_hospitalization'):
            adjusted *= boosters['recent_hospitalization']
        
        if context.get('chronic_conditions', 0) >= 3:
            adjusted *= boosters['multiple_chronic_conditions']
        
        if context.get('high_risk_medication'):
            adjusted *= boosters['high_risk_medication']
        
        if context.get('age', 0) >= 75 and context.get('lives_alone'):
            adjusted *= boosters['elderly_living_alone']
        
        if context.get('prior_adherence_issues'):
            adjusted *= boosters['prior_adherence_issues']
        
        if context.get('days_since_last_contact', 0) > 30:
            adjusted *= boosters['no_recent_contact']
        
        # Apply de-escalation (reduce unnecessary alerts)
        if context.get('successful_contact_last_7_days'):
            adjusted += deescalators['recent_successful_contact']
        
        if context.get('adherence_improving_trend'):
            adjusted += deescalators['improving_trend']
        
        if context.get('medication_count', 10) <= 2:
            adjusted += deescalators['low_complexity_regimen']
        
        # Cap at 1.0
        return min(adjusted, 1.0)
    
    def _determine_priority(self, adjusted_score: float) -> AlertPriority:
        """Determine priority level based on adjusted risk score"""
        thresholds = self.config['thresholds']
        
        if adjusted_score >= thresholds['urgent']:
            return AlertPriority.URGENT
        elif adjusted_score >= thresholds['high']:
            return AlertPriority.HIGH
        elif adjusted_score >= thresholds['medium']:
            return AlertPriority.MEDIUM
        elif adjusted_score >= thresholds['low']:
            return AlertPriority.LOW
        else:
            return AlertPriority.MONITOR
    
    def _exceeds_daily_limit(self, priority: AlertPriority) -> bool:
        """Check if daily alert limit exceeded for this priority"""
        if priority == AlertPriority.MONITOR:
            return False  # No limit on monitoring
        
        today = datetime.now().date()
        
        # Count today's alerts at this priority
        today_count = sum(
            1 for task in self.review_queue
            if datetime.fromisoformat(task['created_at']).date() == today
            and task['priority'] == priority.value
        )
        
        limit = self.config['daily_limits'].get(priority.value, float('inf'))
        return today_count >= limit
    
    def _downgrade_priority(self, priority: AlertPriority) -> AlertPriority:
        """Downgrade priority by one level to manage alert volume"""
        priority_order = [
            AlertPriority.URGENT,
            AlertPriority.HIGH,
            AlertPriority.MEDIUM,
            AlertPriority.LOW,
            AlertPriority.MONITOR
        ]
        
        current_index = priority_order.index(priority)
        if current_index < len(priority_order) - 1:
            return priority_order[current_index + 1]
        return priority
    
    def _calculate_deadline(self, priority: AlertPriority) -> datetime:
        """Calculate review deadline based on priority"""
        if priority == AlertPriority.MONITOR:
            return datetime.now() + timedelta(days=30)
        
        hours = self.config['review_deadlines'].get(priority.value, 168)
        return datetime.now() + timedelta(hours=hours)
    
    def _explain_priority(self, adjusted_score: float, 
                         risk_factors: List[str],
                         context: Optional[Dict]) -> str:
        """Generate human-readable explanation for priority level"""
        explanations = []
        
        # Base risk
        explanations.append(f"AI risk score: {adjusted_score:.2f}")
        
        # Top risk factors
        if risk_factors:
            top_factors = risk_factors[:3]
            explanations.append(f"Key factors: {', '.join(top_factors)}")
        
        # Contextual boosters
        if context:
            if context.get('recent_hospitalization'):
                explanations.append("Recent hospitalization")
            if context.get('high_risk_medication'):
                explanations.append("High-risk medication")
            if context.get('age', 0) >= 75:
                explanations.append("Elderly patient")
        
        return " | ".join(explanations)
    
    def record_clinical_decision(self, task_id: str, 
                                intervention_type: InterventionType,
                                rationale: str,
                                clinician_id: str,
                                override_ai: bool = False,
                                override_reason: Optional[str] = None) -> Dict:
        """
        Record the human clinician's decision on a review task.
        
        This is where HUMAN JUDGMENT is applied and documented.
        
        Args:
            task_id: Review task ID
            intervention_type: Type of intervention decided
            rationale: Clinical rationale for decision
            clinician_id: ID of deciding clinician
            override_ai: Whether clinician overrode AI recommendation
            override_reason: Reason for override (if applicable)
            
        Returns:
            Updated review task with clinical decision
        """
        
        # Find the task
        task = next((t for t in self.review_queue if t['task_id'] == task_id), None)
        
        if not task:
            raise ValueError(f"Review task {task_id} not found")
        
        # Record the decision
        task['clinical_decision'] = {
            'decision_made': True,
            'intervention_type': intervention_type.value,
            'intervention_scheduled': datetime.now().isoformat(),
            'decision_rationale': rationale,
            'decided_by': clinician_id,
            'decided_at': datetime.now().isoformat(),
            'override_ai': override_ai,
            'override_reason': override_reason
        }
        
        task['status'] = 'decision_made'
        task['reviewed_by'] = clinician_id
        task['reviewed_at'] = datetime.now().isoformat()
        
        # Log the intervention
        self.intervention_log.append({
            'task_id': task_id,
            'patient_id': task['patient_id'],
            'ai_risk_score': task['ai_prediction']['risk_score'],
            'priority': task['priority'],
            'intervention_type': intervention_type.value,
            'override_ai': override_ai,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(
            f"Clinical decision recorded: {task_id} | "
            f"Intervention: {intervention_type.value} | "
            f"Clinician: {clinician_id} | "
            f"Override: {override_ai}"
        )
        
        return task
    
    def get_review_queue(self, priority: Optional[AlertPriority] = None,
                        assigned_to: Optional[str] = None) -> List[Dict]:
        """
        Get pending review tasks for clinicians.
        
        Args:
            priority: Filter by priority level
            assigned_to: Filter by assignee
            
        Returns:
            List of review tasks
        """
        tasks = [t for t in self.review_queue if t['status'] == 'pending_review']
        
        if priority:
            tasks = [t for t in tasks if t['priority'] == priority.value]
        
        if assigned_to:
            tasks = [t for t in tasks if t.get('assigned_to') == assigned_to]
        
        # Sort by priority and deadline
        priority_order = {
            'urgent': 0,
            'high': 1,
            'medium': 2,
            'low': 3,
            'monitor': 4
        }
        
        tasks.sort(
            key=lambda x: (
                priority_order[x['priority']],
                datetime.fromisoformat(x['review_deadline'])
            )
        )
        
        return tasks
    
    def get_overdue_tasks(self) -> List[Dict]:
        """Get tasks past their review deadline"""
        now = datetime.now()
        
        overdue = [
            task for task in self.review_queue
            if task['status'] == 'pending_review'
            and datetime.fromisoformat(task['review_deadline']) < now
        ]
        
        return sorted(overdue, key=lambda x: x['review_deadline'])
    
    def generate_escalation_report(self) -> Dict:
        """
        Generate report on escalation patterns and clinical decisions.
        
        Useful for:
        - Quality improvement
        - Understanding AI-human agreement
        - Identifying systematic issues
        """
        
        total_tasks = len(self.review_queue)
        decided_tasks = [t for t in self.review_queue if t['clinical_decision']['decision_made']]
        
        # Calculate agreement rate (clinician followed AI recommendation)
        agreements = sum(1 for t in decided_tasks if not t['clinical_decision']['override_ai'])
        agreement_rate = agreements / len(decided_tasks) if decided_tasks else 0
        
        # Priority distribution
        priority_dist = {}
        for priority in AlertPriority:
            count = sum(1 for t in self.review_queue if t['priority'] == priority.value)
            priority_dist[priority.value] = count
        
        # Intervention types
        intervention_dist = {}
        for task in decided_tasks:
            intervention = task['clinical_decision']['intervention_type']
            intervention_dist[intervention] = intervention_dist.get(intervention, 0) + 1
        
        return {
            'summary': {
                'total_tasks_created': total_tasks,
                'tasks_reviewed': len(decided_tasks),
                'tasks_pending': total_tasks - len(decided_tasks),
                'ai_human_agreement_rate': round(agreement_rate, 3),
                'override_rate': round(1 - agreement_rate, 3)
            },
            'priority_distribution': priority_dist,
            'intervention_distribution': intervention_dist,
            'overdue_tasks': len(self.get_overdue_tasks()),
            'generated_at': datetime.now().isoformat()
        }


class EscalationWorkflow:
    """
    End-to-end workflow showing AI prediction → Human decision separation
    """
    
    def __init__(self):
        self.policy = EscalationPolicy()
    
    def process_ai_prediction(self, patient_id: str, ai_risk_score: float,
                             ai_explanation: List[str],
                             patient_context: Dict) -> Dict:
        """
        Complete workflow from AI prediction to human review task.
        
        DEMONSTRATES THE SEPARATION:
        1. AI makes prediction (risk score + explanation)
        2. Escalation policy creates review task
        3. Human reviews and decides
        4. Human decision is documented
        
        Args:
            patient_id: Patient identifier
            ai_risk_score: AI-predicted risk (0-1)
            ai_explanation: AI's explanation of risk factors
            patient_context: Additional patient information
            
        Returns:
            Review task created for human clinician
        """
        
        logger.info(f"Processing AI prediction for patient {patient_id}")
        logger.info(f"  AI Risk Score: {ai_risk_score:.3f}")
        logger.info(f"  AI Explanation: {ai_explanation}")
        
        # Create review task (NOT an automated intervention)
        review_task = self.policy.create_review_task(
            patient_id=patient_id,
            risk_score=ai_risk_score,
            risk_factors=ai_explanation,
            patient_context=patient_context
        )
        
        logger.info(
            f"  → Review task created: {review_task['task_id']}\n"
            f"  → Priority: {review_task['priority']}\n"
            f"  → Assigned for human review by: {review_task['review_deadline']}\n"
            f"  → AI DOES NOT DIRECTLY CONTACT PATIENT"
        )
        
        return review_task
    
    def simulate_clinician_review(self, task_id: str, 
                                 clinician_id: str = "DR_SMITH") -> Dict:
        """
        Simulate a clinician reviewing and making a decision.
        
        In production, this would be done through a UI where
        clinicians review AI predictions and make their own judgments.
        """
        
        # Get the task
        task = next(
            (t for t in self.policy.review_queue if t['task_id'] == task_id),
            None
        )
        
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Simulate clinician decision-making
        ai_score = task['ai_prediction']['adjusted_score']
        
        # Clinician might agree with AI or override
        if ai_score >= 0.7:
            # High risk - clinician decides to call
            intervention = InterventionType.PHONE_CALL
            rationale = (
                "Agree with AI assessment. Multiple concerning factors "
                "warrant direct phone contact to assess patient status."
            )
            override = False
        elif ai_score >= 0.5:
            # Medium risk - pharmacist consult
            intervention = InterventionType.PHARMACIST_CONSULT
            rationale = (
                "Moderate risk. Pharmacist consult appropriate to review "
                "medication regimen and adherence barriers."
            )
            override = False
        else:
            # Clinician overrides - sees context AI missed
            intervention = InterventionType.NO_ACTION
            rationale = (
                "Patient recently had successful in-person visit. "
                "Adherence confirmed. Will continue routine monitoring."
            )
            override = True
        
        # Record the decision
        return self.policy.record_clinical_decision(
            task_id=task_id,
            intervention_type=intervention,
            rationale=rationale,
            clinician_id=clinician_id,
            override_ai=override
        )


# Example usage demonstrating the separation
if __name__ == "__main__":
    
    print("=" * 70)
    print("DEMONSTRATION: AI Prediction → Human Decision Separation")
    print("=" * 70)
    
    workflow = EscalationWorkflow()
    
    # Scenario 1: High-risk patient
    print("\n📊 SCENARIO 1: AI detects high-risk patient")
    print("-" * 70)
    
    review_task = workflow.process_ai_prediction(
        patient_id="PT00123",
        ai_risk_score=0.82,
        ai_explanation=[
            "Missed 70% of IVR calls in past 2 weeks",
            "SMS read rate dropped to 20%",
            "Prescription refill 5 days overdue"
        ],
        patient_context={
            'age': 72,
            'chronic_conditions': 3,
            'high_risk_medication': True,
            'recent_hospitalization': True
        }
    )
    
    print(f"\n✅ Review Task Created (NOT automated intervention):")
    print(f"   Task ID: {review_task['task_id']}")
    print(f"   Priority: {review_task['priority'].upper()}")
    print(f"   Review By: {review_task['review_deadline'][:16]}")
    print(f"   Status: Awaiting HUMAN clinician review")
    
    # Simulate clinician review
    print(f"\n👨‍⚕️ Clinician reviews the task and makes decision...")
    decision = workflow.simulate_clinician_review(review_task['task_id'])
    
    print(f"\n✅ Clinical Decision Recorded:")
    print(f"   Intervention: {decision['clinical_decision']['intervention_type']}")
    print(f"   Rationale: {decision['clinical_decision']['decision_rationale']}")
    print(f"   Decided by: {decision['clinical_decision']['decided_by']}")
    
    # Generate report
    print("\n" + "=" * 70)
    print("ESCALATION REPORT")
    print("=" * 70)
    
    report = workflow.policy.generate_escalation_report()
    print(f"\nTotal Review Tasks: {report['summary']['total_tasks_created']}")
    print(f"Tasks Reviewed: {report['summary']['tasks_reviewed']}")
    print(f"AI-Human Agreement: {report['summary']['ai_human_agreement_rate']*100:.1f}%")
    print(f"\nPriority Distribution:")
    for priority, count in report['priority_distribution'].items():
        if count > 0:
            print(f"  {priority.capitalize()}: {count}")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("  ✓ AI predicts risk scores")
    print("  ✓ System creates review tasks")
    print("  ✓ HUMANS make final decisions")
    print("  ✓ AI NEVER directly triggers patient contact")
    print("=" * 70)
