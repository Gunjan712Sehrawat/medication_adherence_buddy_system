# Model Architecture Documentation

## Overview

The Medication Adherence Risk Scoring System uses a hybrid ensemble approach combining rule-based clinical logic with machine learning for optimal interpretability and performance.

## Architecture Components

### 1. Rule-Based Engine (`rule_engine.py`)

**Purpose**: Provide clinically interpretable, explainable risk scores based on established medical knowledge.

**Key Features**:
- 12+ clinically-validated risk rules
- Weighted by evidence strength
- Categories: IVR, SMS, Behavioral, Demographics
- Works with incomplete data
- Always provides explanations

**Rule Categories**:

#### IVR Rules
- `high_ivr_miss_rate`: >60% missed calls → Score 0.8
- `no_ivr_response`: Zero response to 3+ calls → Score 0.9
- `declining_ivr_response`: Deteriorating pattern → Score 0.7

#### SMS Rules
- `low_sms_read_rate`: <40% read rate → Score 0.6
- `slow_sms_response`: >24hr avg response → Score 0.5
- `no_sms_engagement`: No reads after 5+ messages → Score 0.85

#### Behavioral Rules
- `missed_refill_window`: >3 days overdue → Score 0.9
- `early_treatment_phase`: First 2 weeks → Score 0.4
- `weekend_nonadherence`: Low weekend adherence → Score 0.5

#### Demographic Rules
- `high_risk_age`: <25 or >75 years → Score 0.3
- `multiple_medications`: 5+ medications → Score 0.4
- `complex_regimen`: 3+ doses daily → Score 0.35

**Scoring Logic**:
```
risk_score = Σ(max_score_per_category × category_weight)

Category Weights:
- IVR: 35%
- SMS: 30%
- Behavioral: 25%
- Demographics: 10%
```

### 2. Machine Learning Component (`ensemble.py`)

**Purpose**: Capture complex, non-linear patterns that rules may miss.

**Model**: LightGBM Gradient Boosting

**Why LightGBM**:
- Fast training on small datasets
- Built-in handling of missing values
- Native support for categorical features
- Low memory footprint
- Excellent performance

**Hyperparameters** (tuned for low-data regime):
```yaml
n_estimators: 100
max_depth: 5          # Prevent overfitting
learning_rate: 0.05   # Conservative learning
min_child_samples: 20 # Avoid small leaf nodes
subsample: 0.8        # Row sampling
colsample_bytree: 0.8 # Feature sampling
```

**Feature Engineering**:
- Derived rates (response rate, miss rate)
- Temporal features (early/established treatment)
- Interaction features (overall engagement)
- Missing value imputation

### 3. Ensemble Combination

**Strategy**: Weighted average of rule-based and ML predictions

```
final_score = 0.4 × rule_score + 0.6 × ml_score
```

**Rationale**:
- Rules provide interpretability and domain knowledge
- ML captures data patterns and interactions
- Ensemble is more robust than either alone

**Calibration**:
- Isotonic regression for well-calibrated probabilities
- Ensures risk scores match actual risk rates

## Explainability

### SHAP (SHapley Additive exPlanations)

Used to explain ML predictions:
- Feature importance globally
- Individual prediction explanations
- Interaction effects

### Combined Explanation

For each prediction:
1. Overall risk score (0-1)
2. Risk level (LOW/MEDIUM/HIGH)
3. Rule-based factors (human-readable)
4. ML feature contributions (SHAP values)
5. Recommended interventions

## Handling Low-Data Scenarios

### Strategies:

1. **Synthetic Data Generation**
   - Realistic correlations
   - Controlled class balance
   - Augmentation with noise

2. **Regularization**
   - Max depth limit
   - Min samples per leaf
   - L2 regularization

3. **Cross-Validation**
   - Stratified 5-fold CV
   - Prevents overfitting
   - Robust performance estimates

4. **Transfer Learning** (future)
   - Pre-trained on large public datasets
   - Fine-tune on institution data

## Privacy Preservation

### Techniques:

1. **Data Anonymization**
   - Patient ID hashing (SHA-256)
   - PII removal/encryption

2. **Differential Privacy** (optional)
   - DP-SGD training
   - Noise injection
   - Privacy budget (ε=1.0)

3. **Federated Learning** (future)
   - Train on decentralized data
   - No data sharing between institutions

## Model Evaluation

### Metrics:

- **ROC-AUC**: Overall discrimination ability
- **Precision**: Minimize false positives (alert fatigue)
- **Recall**: Catch most at-risk patients
- **Calibration**: Risk scores match actual rates

### Threshold Tuning:

Optimize based on institution priorities:
- High recall: Catch all at-risk (more alerts)
- High precision: Only high-confidence (fewer alerts)
- Balanced: F1 score optimization

## Production Deployment

### Model Serving:

1. **REST API** (Flask)
   - Single prediction: `/api/predict`
   - Batch prediction: `/api/predict_batch`
   - Explanation: `/api/explain`

2. **Dashboard** (Streamlit)
   - Interactive risk assessment
   - Model training interface
   - Analytics and monitoring

### Monitoring:

Track in production:
- Prediction latency
- Model drift (feature distributions)
- Alert rates
- Intervention outcomes

### Retraining:

When to retrain:
- New data available (monthly)
- Performance degradation detected
- Population characteristics change
- New medication programs

## Future Enhancements

1. **Deep Learning** (if data allows)
   - LSTM for temporal patterns
   - Attention mechanisms

2. **Multi-Task Learning**
   - Predict adherence + outcomes
   - Shared representations

3. **Causal Inference**
   - Estimate intervention effects
   - Optimal treatment assignment

4. **Active Learning**
   - Query most informative cases
   - Efficient labeling

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Clinical Adherence Literature](https://www.ncbi.nlm.nih.gov/)
