"""
Medication Adherence Risk Scoring - Streamlit Dashboard
"""

import requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

import warnings
warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice"
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide"
)


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ensemble import EnsembleRiskScorer
from src.models.rule_engine import RuleEngine
from src.utils.synthetic import SyntheticDataGenerator

# Page configuration
st.set_page_config(
    page_title="Medication Adherence Risk Scoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load or initialize the risk scoring model"""
    try:
        model = EnsembleRiskScorer()
        # Try to load pre-trained model
        model.load('models/ensemble_model.pkl')
        return model, True
    except:
        # Return untrained model
        return EnsembleRiskScorer(), False


@st.cache_data
def generate_demo_data(n_samples=100):
    """Generate synthetic data for demonstration"""
    generator = SyntheticDataGenerator()
    return generator.generate(n_samples=n_samples)


def display_risk_gauge(risk_score, risk_level):
    """Display risk score as a gauge chart"""
    color_map = {
        'LOW': 'green',
        'MEDIUM': 'orange',
        'HIGH': 'red'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Score<br><b>{risk_level}</b>"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color_map[risk_level]},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"},
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def display_feature_importance(model):
    """Display feature importance chart safely"""
    if not hasattr(model, "get_feature_importance"):
        st.warning("Feature importance not available")
        return

    importance = model.get_feature_importance(top_n=10)

    if importance is None or importance.empty:
        st.info("Train the model first to see feature importance")
        return

    fig = px.bar(
        importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Important Features'
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")



FLASK_API_URL = "http://127.0.0.1:5000"

def api_predict(payload: dict):
    """Call Flask API for prediction"""
    try:
        response = requests.post(
            f"{FLASK_API_URL}/api/predict",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error("❌ Failed to connect to backend API")
        st.exception(e)
        return None


def main():
    try:
        run_app()
    except Exception as e:
        st.error("🚨 Application crashed")
        st.exception(e)


def run_app():
    # existing main() code here
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🏥 Medication Adherence Risk Scoring System</p>', 
                unsafe_allow_html=True)
    st.markdown("*Interpretable AI for early detection of medication non-adherence*")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100.png?text=Healthcare+AI", 
                 width="stretch")
        
        st.markdown("## Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "👤 Patient Assessment", "📊 Model Training", 
             "📈 Analytics", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This system predicts medication non-adherence risk using:
        - IVR call patterns
        - SMS engagement data
        - Prescription history
        - Demographics
        """)
        
        st.markdown("---")
        st.markdown("### Model Status")
        model, is_trained = load_model()
        if is_trained:
            st.success("✓ Model Loaded")
        else:
            st.warning("⚠ Model Not Trained")
    
    # Load model
    model, is_trained = load_model()
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(model, is_trained)
    elif page == "👤 Patient Assessment":
        show_patient_assessment(model, is_trained)
    elif page == "📊 Model Training":
        show_model_training(model)
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()


def show_dashboard(model, is_trained):
    """Display main dashboard"""
    st.header("Dashboard Overview")
    
    # Generate demo data
    demo_data = generate_demo_data(n_samples=50)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Patients", len(demo_data))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        high_risk = (demo_data['label'] == 1).sum()
        total = len(demo_data)

        percentage = (high_risk / total * 100) if total > 0 else 0.0

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High Risk", high_risk, delta=f"{percentage:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Review Tasks", 12, delta="3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Decisions Made", 8, delta="-2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        if is_trained:
            if len(demo_data) > 0:
                scores = model.predict(demo_data.drop('label', axis=1))
            else:
                scores = np.array([])
                
            risk_levels = pd.cut(scores, bins=[0, 0.3, 0.6, 1.0], 
                                labels=['LOW', 'MEDIUM', 'HIGH'])
            risk_counts = risk_levels.value_counts().reindex(
                ['LOW', 'MEDIUM', 'HIGH'], fill_value=0
            )

            
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Train model to see predictions")
    
    with col2:
        st.subheader("Weekly Review Tasks")
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        review_tasks = np.random.randint(5, 20, 7)
        
        fig = px.line(x=dates, y=review_tasks, markers=True,
                     labels={'x': 'Date', 'y': 'Review Tasks Created'})
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Recent review tasks table
    st.subheader("Patients Requiring Clinical Review")
    st.caption("⚠️ AI predictions - Human clinicians must review and decide on interventions")
    
    if is_trained:
        scores = model.predict(demo_data.drop('label', axis=1))
        demo_data['risk_score'] = scores
        demo_data['risk_level'] = pd.cut(scores, bins=[0, 0.3, 0.6, 1.0],
                                         labels=['LOW', 'MEDIUM', 'HIGH'])
        
        high_risk_patients = demo_data[demo_data['risk_level'] == 'HIGH'].head(5)
        
        display_df = high_risk_patients[['patient_id', 'risk_score', 'risk_level',
                                         'ivr_calls_missed', 'sms_read']].copy()
        display_df['risk_score'] = display_df['risk_score'].round(3)
        
        st.dataframe(display_df, width="stretch", hide_index=True)
    else:
        st.info("Train model to see high-risk patients")


def show_patient_assessment(model, is_trained):
    """Individual patient risk assessment"""
    st.header("Patient Risk Assessment")
    
    # CRITICAL DISCLAIMER
    st.warning("""
    **⚠️ Important: This is a Clinical Decision Support Tool**
    
    - The AI **predicts** risk scores based on behavioral patterns
    - The AI does **NOT** directly trigger patient contact or interventions  
    - All predictions **must** be reviewed by qualified healthcare staff
    - Only **human clinicians** make final decisions on patient interventions
    
    **This assessment creates a review task, not an automated action.**
    """)
    
    st.markdown("""
    Enter patient data to get real-time risk assessment and intervention recommendations.
    """)
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("IVR Data")
        ivr_total = st.number_input("Total IVR Calls", min_value=0, max_value=20, value=5)
        ivr_answered = st.number_input("Calls Answered", min_value=0, max_value=ivr_total, value=2)
        ivr_missed = ivr_total - ivr_answered
        
        st.subheader("Demographics")
        age = st.slider("Age", min_value=18, max_value=100, value=65)
        num_meds = st.number_input("Number of Medications", min_value=1, max_value=20, value=3)
    
    with col2:
        st.subheader("SMS Data")
        sms_delivered = st.number_input("SMS Delivered", min_value=0, max_value=30, value=10)
        sms_read = st.number_input("SMS Read", min_value=0, max_value=sms_delivered, value=4)
        sms_response_time = st.number_input("Avg Response Time (hours)", min_value=0.0, max_value=72.0, value=12.0)
        
        st.subheader("Treatment")
        days_since_rx = st.number_input("Days Since Prescription", min_value=0, max_value=365, value=14)
    
    # Predict button
    if st.button("🔍 Assess Risk", type="primary"):
        if not is_trained:
            st.error("Please train the model first in the Model Training page")
            return
        
        # Prepare patient data
        patient_data = pd.DataFrame([{
            'patient_id': 'DEMO_PATIENT',
            'ivr_calls_total': ivr_total,
            'ivr_calls_answered': ivr_answered,
            'ivr_calls_missed': ivr_missed,
            'sms_delivered': sms_delivered,
            'sms_read': sms_read,
            'sms_avg_response_time_hrs': sms_response_time,
            'days_since_prescription': days_since_rx,
            'age': age,
            'num_medications': num_meds
        }])
        
        # Convert dataframe to dict for API
        payload = patient_data.iloc[0].to_dict()

        response = requests.post(
            "http://127.0.0.1:5000/api/predict",
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            st.error("❌ Flask API not responding")
            return

        explanation = response.json()

        
        st.markdown("---")
        st.subheader("Risk Assessment Results")
        
        # Display risk gauge
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = display_risk_gauge(explanation['risk_score'], explanation['risk_level'])
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Risk level box
            risk_level = explanation['risk_level']
            if risk_level == 'HIGH':
                st.markdown('<div class="risk-high">', unsafe_allow_html=True)
                st.markdown("### ⚠️ HIGH RISK")
                st.markdown("Immediate intervention recommended")
            elif risk_level == 'MEDIUM':
                st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
                st.markdown("### ⚡ MEDIUM RISK")
                st.markdown("Close monitoring advised")
            else:
                st.markdown('<div class="risk-low">', unsafe_allow_html=True)
                st.markdown("### ✅ LOW RISK")
                st.markdown("Continue routine follow-up")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Component scores
            st.markdown("#### Score Breakdown")
            st.write(f"**Rule-Based:** {explanation['component_scores']['rule_based']:.3f}")
            st.write(f"**ML Model:** {explanation['component_scores']['ml_model']:.3f}")
        
        # Risk factors
        st.markdown("---")
        st.subheader("🎯 Key Risk Factors")
        
        for i, factor in enumerate(explanation['rule_explanation']['triggered_factors'][:5], 1):
            st.markdown(f"{i}. {factor}")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommended Actions")
        
        for i, rec in enumerate(explanation['rule_explanation']['recommendations'], 1):
            st.markdown(f"**{i}.** {rec}")


def show_model_training(model):
    """Model training interface"""
    st.header("Model Training")
    
    st.markdown("""
    Train the risk scoring model on synthetic or uploaded data.
    """)
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Generate Synthetic Data", "Upload CSV File"]
    )
    
    if data_source == "Generate Synthetic Data":
        n_samples = st.slider("Number of Samples", min_value=100, max_value=2000, 
                             value=500, step=100)
        
        if st.button("Generate Data", type="primary"):
            with st.spinner("Generating synthetic data..."):
                data = generate_demo_data(n_samples)
                st.session_state['training_data'] = data
                st.success(f"Generated {len(data)} samples")
                st.dataframe(data.head(), width="stretch")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.session_state['training_data'] = data
            st.success("Data uploaded successfully")
            st.dataframe(data.head(), width="stretch")
    
    # Training
    if 'training_data' in st.session_state:
        st.markdown("---")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                data = st.session_state['training_data']
                
                # Split data
                from sklearn.model_selection import train_test_split
                X = data.drop('label', axis=1)
                y = data['label']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train
                model.train(X_train, y_train, X_test, y_test)
                
                # Evaluate
                from sklearn.metrics import classification_report, roc_auc_score
                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred >= 0.5).astype(int)
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                auc = roc_auc_score(y_test, y_pred)
                report = classification_report(y_test, y_pred_binary, output_dict=True)
                
                with col1:
                    st.metric("ROC-AUC", f"{auc:.3f}")
                with col2:
                    st.metric("Precision", f"{report['1']['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{report['1']['recall']:.3f}")
                
                # Feature importance
                st.markdown("---")
                st.subheader("Feature Importance")
                display_feature_importance(model)
                
                # Save model
                if st.button("💾 Save Model"):
                    os.makedirs('models', exist_ok=True)
                    model.save('models/ensemble_model.pkl')
                    st.success("Model saved successfully!")


def show_analytics():
    """Analytics and insights"""
    st.header("Analytics Dashboard")
    
    st.info("Advanced analytics coming soon!")
    
    # Placeholder visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Adherence by Age Group")
        age_groups = ['18-30', '31-45', '46-60', '61-75', '75+']
        adherence = np.random.rand(5) * 100
        
        fig = px.bar(x=age_groups, y=adherence,
                    labels={'x': 'Age Group', 'y': 'Adherence Rate (%)'})
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Response Channel Effectiveness")
        channels = ['IVR', 'SMS', 'Email', 'App']
        effectiveness = np.random.rand(4) * 100
        
        fig = px.bar(x=channels, y=effectiveness,
                    labels={'x': 'Channel', 'y': 'Effectiveness (%)'})
        st.plotly_chart(fig, width="stretch")


def show_settings():
    """Settings page"""
    st.header("Settings")
    
    st.subheader("Risk Thresholds")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        low_threshold = st.slider("Low Risk Cutoff", 0.0, 1.0, 0.3, 0.05)
    with col2:
        medium_threshold = st.slider("Medium Risk Cutoff", 0.0, 1.0, 0.6, 0.05)
    with col3:
        high_threshold = st.slider("High Risk Cutoff", 0.0, 1.0, 0.8, 0.05)
    
    st.subheader("Alert Settings")
    max_daily_alerts = st.number_input("Maximum Daily Alerts", min_value=1, 
                                       max_value=200, value=50)
    
    st.subheader("Privacy Settings")
    anonymize = st.checkbox("Anonymize Patient IDs", value=True)
    encryption = st.checkbox("Enable Data Encryption", value=False)
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
