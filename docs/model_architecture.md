Model Architecture

**1. Architectural Scope**

This document describes the internal technical architecture of the Medication Adherence Buddy System, focusing on:
 - Data flow between modules
 - Model composition and execution order
 - Scoring logic and aggregation
 - Interfaces between ML, rules, API, and UI layers
Non-architectural topics such as motivation, impact, ethics, and deployment strategy are intentionally excluded.

**2. High-Level Architecture Layout**

The system follows a **layered and modular architecture, separating concerns across data processing, scoring logic, orchestration, and presentation**.

Input Data
(IVR / SMS / Patient Signals)
        │
        ▼
Feature Processing Layer
(src/features, src/utils)
        │
        ├──► Rule-Based Scoring Engine
        │     (deterministic logic)
        │
        ├──► Machine Learning Model
        │     (LightGBM ensemble)
        │
        ▼
Ensemble Risk Aggregator
(src/models/ensemble.py)
        │
        ▼
Explanation Generator
(rule triggers + score breakdown)
        │
        ├──► Flask API (src/api)
        └──► Streamlit UI (streamlit_app)

Each layer can be executed independently and tested in isolation.

**3. Feature Processing Layer**

 - Transform raw engagement signals into numeric features
 - Ensure consistency between training and inference
 - Handle missing or partial inputs

   *Processing Characteristics*
      - Ratio-based features (e.g., read rates, miss rates)
      - Time-based features (e.g., days since prescription)
      - Simple statistical transformations
      - Defensive handling of empty or null values

No learning occurs at this layer. All transformations are deterministic.

**4. Rule-Based Scoring Engine**
   *Location* : src/models (rule logic invoked via ensemble)

   *Role in Architecture*
   - The rule engine provides a deterministic risk estimate based on predefined conditions. It runs before the ML model and produces:   
         - A numeric rule-based risk score
         - A list of triggered rule explanations

   *Architectural Properties*
      - Stateless
      - Order-independent rules
      - Category-based aggregation

   *Rule Grouping*
   Rules are grouped into logical categories such as:
      - IVR engagement
      - SMS engagement
      - Treatment behavior
      - Patient complexity indicators

   For each category:
      - Only the maximum contributing rule is used
      - Prevents score inflation from correlated conditions
      - Category outputs are combined using fixed weights.

**5. Machine Learning Component**
   *Location* : src/models/ensemble.py
               models/ensemble_model.pkl

   *Model Type* : Gradient Boosted Decision Trees (LightGBM)

   *Architectural Role*
      - Learn non-linear interactions between features
      - Capture patterns not explicitly encoded in rules

   *Design Constraints*
      - Operates on the same feature schema as the rule engine
      - Accepts missing values without preprocessing failure
      - Produces a single continuous risk score

   The ML model is read-only at inference time and loaded from a serialized artifact.

**6. Ensemble Risk Aggregation**
   *Location* :  src/models/ensemble.py

   *Purpose*
      Combine outputs from:   - Rule-based engine
                              - ML model

   *Combination Logic*
      - A weighted linear aggregation is applied:  final_risk_score = (rule_score × rule_weight) + (ml_score × ml_weight)
      - Weights are fixed and defined in code to ensure reproducibility.
      - This layer acts as the single source of truth for final risk computation.

**7. Risk Level Mapping**

The continuous risk score is mapped into discrete categories:
   - LOW
   - MEDIUM
   - HIGH
     
This mapping is threshold-based and implemented centrally to ensure consistent interpretation across **API and UI layers**.

**8. Explanation Assembly**
   *Inputs*
      - Triggered rules
      - Rule category contributions
      - Final ensemble score

   *Outputs*
      - Human-readable explanation object
      - Score breakdown by source
      - Trigger summaries

No post-hoc model interpretation libraries are required at runtime; explanations are derived from architectural components directly.

**9. API Integration Layer**
   *Location* : src/api/flask_app.py

   - Load ensemble model at startup
   - Expose prediction and explanation endpoints
   - Validate input schema
   - Return structured JSON responses

The API layer does not perform feature engineering logic directly; it delegates computation to the model layer.

**10. Streamlit Interface Layer**
   *Location* : streamlit_app/app.py

   - Collect user input
   - Display scores, risk levels, and explanations
   - Visualize distributions and summaries
   - The UI consumes either: Direct model calls (local mode), or API responses (service mode)

It contains no business logic related to scoring.

**11. Model Lifecycle Boundaries**

   - Training logic is isolated from inference paths
   - Serialized models are stored outside source code
   - Architecture supports replacement of the ML model without modifying rule logic
   - Rule logic can be updated independently of retraining

**12. Architectural Guarantees**

   - Deterministic behavior for identical inputs
   - Clear separation between logic, orchestration, and presentation
   - No hidden state across requests
   - Human-interpretable scoring pipeline

**13. Non-Goals of the Architecture**

This architecture explicitly does not:

   - Perform automated clinical actions
   - Store personal patient identifiers
   - Self-update or retrain in production
   - Replace human decision-making

**14. Architectural Summary**

The system architecture is a hybrid, layered design where:

   - Rules ensure interpretability and safety
   - ML improves pattern recognition
   - Ensemble logic balances both
   - API and UI layers remain thin and decoupled

This structure enables controlled experimentation, clear debugging, and responsible use in healthcare-adjacent environments.
