"""
Flask REST API for Medication Adherence Risk Scoring
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ensemble import EnsembleRiskScorer
from src.models.rule_engine import RuleEngine
from src.utils.synthetic import SyntheticDataGenerator
from src.escalation.escalation_policy import EscalationWorkflow, AlertPriority

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model = None
rule_engine = RuleEngine()
escalation_workflow = EscalationWorkflow()


def load_model():
    """Load the trained model"""
    global model
    try:
        model = EnsembleRiskScorer()
        model.load('models/ensemble_model.pkl')
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.info("Using rule-based engine only")
        model = None
        return False


# Load model on startup
model_loaded = load_model()


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Medication Adherence Risk Scoring API',
        'version': '1.0.0',
        'status': 'operational',
        'model_loaded': model_loaded,
        'important': 'AI predicts risk. HUMANS make intervention decisions.',
        'endpoints': {
            # AI Prediction Endpoints
            'health': '/api/health',
            'predict': '/api/predict',
            'predict_batch': '/api/predict_batch',
            'explain': '/api/explain',
            'rules_only': '/api/rules_only',
            
            # Escalation & Review Endpoints (AI → Human separation)
            'create_review_task': '/api/create_review_task',
            'review_queue': '/api/review_queue',
            'record_decision': '/api/record_decision',
            
            # Utilities
            'demo_data': '/api/demo_data'
        },
        'workflow': (
            '1. AI predicts risk score → '
            '2. System creates review task → '
            '3. Human reviews → '
            '4. Human decides intervention'
        )
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict risk score for a single patient
    
    Request body:
    {
        "patient_id": "PT12345",
        "ivr_calls_total": 5,
        "ivr_calls_answered": 2,
        "ivr_calls_missed": 3,
        "sms_delivered": 10,
        "sms_read": 4,
        "sms_avg_response_time_hrs": 12.5,
        "age": 65,
        "num_medications": 3,
        "days_since_prescription": 14
    }
    """
    try:
        # Get patient data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['patient_id']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([data])
        
        # Predict using ensemble model or rules only
        if model_loaded and model is not None:
            risk_score = float(model.predict(patient_df)[0])
            prediction_method = 'ensemble'
        else:
            risk_score, _ = rule_engine.predict(data)
            prediction_method = 'rule_based'
        
        # Categorize risk
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Get rule-based explanation
        _, explanations = rule_engine.predict(data)
        
        # IMPORTANT: This creates a REVIEW TASK, not an automated intervention
        # Human clinicians must review and decide on actual interventions
        review_task_created = risk_score >= 0.5  # Only create tasks for medium+ risk
        
        response = {
            'patient_id': data['patient_id'],
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            
            # CRITICAL: Changed from 'intervention_required' to 'review_task_created'
            # AI does NOT trigger interventions - it creates review tasks for humans
            'review_task_created': review_task_created,
            'requires_human_review': review_task_created,
            
            'prediction_method': prediction_method,
            'explanation': {
                'top_factors': explanations[:3],
                'all_factors': explanations
            },
            
            # Clear disclaimer
            'disclaimer': 'This is a risk prediction for clinical review. '
                         'A qualified healthcare professional must review and '
                         'make final decisions on patient interventions.',
            
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction for {data['patient_id']}: {risk_score:.3f} ({risk_level})")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict risk scores for multiple patients
    
    Request body:
    {
        "patients": [
            { patient data },
            { patient data },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        patients = data['patients']
        
        if not patients:
            return jsonify({'error': 'Empty patient list'}), 400
        
        # Convert to DataFrame
        patients_df = pd.DataFrame(patients)
        
        # Predict
        if model_loaded and model is not None:
            risk_scores = model.predict(patients_df)
            prediction_method = 'ensemble'
        else:
            # Use rule engine for batch
            results = rule_engine.predict_batch(patients_df)
            risk_scores = results['risk_score'].values
            prediction_method = 'rule_based'
        
        # Prepare response
        predictions = []
        for i, (idx, row) in enumerate(patients_df.iterrows()):
            score = float(risk_scores[i])
            
            if score >= 0.7:
                risk_level = "HIGH"
            elif score >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            predictions.append({
                'patient_id': row.get('patient_id', f'patient_{i}'),
                'risk_score': round(score, 3),
                'risk_level': risk_level,
                'intervention_required': score >= 0.7
            })
        
        response = {
            'total_patients': len(predictions),
            'high_risk_count': sum(1 for p in predictions if p['risk_level'] == 'HIGH'),
            'predictions': predictions,
            'prediction_method': prediction_method,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction for {len(predictions)} patients")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """
    Get detailed explanation for a prediction
    
    Request body: Same as /api/predict
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get rule-based explanation
        rule_explanation = rule_engine.explain_prediction(data)
        
        # Get ML explanation if model is loaded
        if model_loaded and model is not None:
            patient_df = pd.DataFrame([data])
            ml_explanation = model.explain(patient_df, patient_idx=0)
            
            response = {
                'patient_id': data.get('patient_id', 'unknown'),
                'risk_assessment': ml_explanation,
                'timestamp': datetime.now().isoformat()
            }
        else:
            response = {
                'patient_id': data.get('patient_id', 'unknown'),
                'risk_assessment': rule_explanation,
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rules_only', methods=['POST'])
def rules_only():
    """
    Get risk score using only rule-based engine (no ML)
    Useful for comparison or when ML model is not available
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get rule-based prediction
        risk_score, explanations = rule_engine.predict(data)
        
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        response = {
            'patient_id': data.get('patient_id', 'unknown'),
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'intervention_required': risk_score >= 0.7,
            'triggered_rules': explanations,
            'method': 'rule_based_only',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Rule-based prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo_data', methods=['GET'])
def get_demo_data():
    """Generate synthetic demo data"""
    try:
        n_samples = request.args.get('n', default=10, type=int)
        n_samples = min(n_samples, 100)  # Limit to 100
        
        generator = SyntheticDataGenerator()
        data = generator.generate(n_samples=n_samples)
        
        return jsonify({
            'count': len(data),
            'data': data.to_dict('records')
        }), 200
    
    except Exception as e:
        logger.error(f"Demo data generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/create_review_task', methods=['POST'])
def create_review_task():
    """
    Create a review task for human clinicians based on AI prediction.
    
    IMPORTANT: This endpoint demonstrates the separation between
    AI prediction and human decision-making.
    
    The AI predicts risk, the system creates a review task,
    but ONLY humans decide on actual patient interventions.
    
    Request body:
    {
        "patient_id": "PT12345",
        "ivr_calls_total": 5,
        "ivr_calls_answered": 2,
        ...
        "patient_context": {
            "age": 72,
            "chronic_conditions": 3,
            "recent_hospitalization": true
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'patient_id' not in data:
            return jsonify({'error': 'patient_id required'}), 400
        
        # Get AI prediction
        patient_df = pd.DataFrame([data])
        
        if model_loaded and model is not None:
            risk_score = float(model.predict(patient_df)[0])
            prediction_method = 'ensemble'
        else:
            risk_score, _ = rule_engine.predict(data)
            prediction_method = 'rule_based'
        
        # Get explanation
        _, explanations = rule_engine.predict(data)
        
        # Extract patient context (if provided)
        patient_context = data.get('patient_context', {})
        
        # Create review task using escalation workflow
        review_task = escalation_workflow.process_ai_prediction(
            patient_id=data['patient_id'],
            ai_risk_score=risk_score,
            ai_explanation=explanations,
            patient_context=patient_context
        )
        
        response = {
            'success': True,
            'message': 'Review task created for clinical staff',
            'review_task': {
                'task_id': review_task['task_id'],
                'patient_id': review_task['patient_id'],
                'priority': review_task['priority'],
                'review_deadline': review_task['review_deadline'],
                'ai_prediction': review_task['ai_prediction'],
                'status': review_task['status']
            },
            'important_note': (
                'This is a REVIEW TASK for human clinicians. '
                'The AI has NOT triggered any patient intervention. '
                'A qualified healthcare professional must review this '
                'task and make the final decision on patient contact.'
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"Review task created: {review_task['task_id']} | "
            f"Priority: {review_task['priority']} | "
            f"Patient: {data['patient_id']}"
        )
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Review task creation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/review_queue', methods=['GET'])
def get_review_queue():
    """
    Get pending review tasks for clinicians.
    
    Query parameters:
    - priority: Filter by priority (urgent, high, medium, low, monitor)
    - assigned_to: Filter by assignee
    """
    try:
        priority_param = request.args.get('priority')
        assigned_to = request.args.get('assigned_to')
        
        # Convert priority string to enum if provided
        priority = None
        if priority_param:
            try:
                priority = AlertPriority(priority_param.lower())
            except ValueError:
                return jsonify({
                    'error': f'Invalid priority: {priority_param}',
                    'valid_priorities': [p.value for p in AlertPriority]
                }), 400
        
        # Get queue
        tasks = escalation_workflow.policy.get_review_queue(
            priority=priority,
            assigned_to=assigned_to
        )
        
        return jsonify({
            'count': len(tasks),
            'tasks': tasks,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Review queue error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/record_decision', methods=['POST'])
def record_clinical_decision():
    """
    Record a clinician's decision on a review task.
    
    This endpoint is where HUMAN JUDGMENT is applied and documented.
    
    Request body:
    {
        "task_id": "REVIEW_PT12345_20260205_100000",
        "intervention_type": "phone_call",
        "rationale": "Patient shows multiple risk factors. Will call to assess.",
        "clinician_id": "DR_SMITH",
        "override_ai": false
    }
    """
    try:
        data = request.get_json()
        
        required = ['task_id', 'intervention_type', 'rationale', 'clinician_id']
        missing = [f for f in required if f not in data]
        
        if missing:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing
            }), 400
        
        # Import InterventionType enum
        from src.escalation.escalation_policy import InterventionType
        
        # Validate intervention type
        try:
            intervention = InterventionType(data['intervention_type'].lower())
        except ValueError:
            return jsonify({
                'error': f'Invalid intervention type: {data["intervention_type"]}',
                'valid_types': [t.value for t in InterventionType]
            }), 400
        
        # Record the decision
        updated_task = escalation_workflow.policy.record_clinical_decision(
            task_id=data['task_id'],
            intervention_type=intervention,
            rationale=data['rationale'],
            clinician_id=data['clinician_id'],
            override_ai=data.get('override_ai', False),
            override_reason=data.get('override_reason')
        )
        
        response = {
            'success': True,
            'message': 'Clinical decision recorded',
            'task_id': updated_task['task_id'],
            'decision': updated_task['clinical_decision'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"Decision recorded: {data['task_id']} | "
            f"Intervention: {data['intervention_type']} | "
            f"Clinician: {data['clinician_id']}"
        )
        
        return jsonify(response), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Decision recording error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask API on port {port}")
    logger.info(f"Model loaded: {model_loaded}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
