# 🏥 Medication Adherence Risk Scoring Buddy System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)

An AI-powered **clinical decision support prototype** that helps identify patients who may be at risk of **medication non-adherence**, using limited IVR and SMS engagement data.

The system assists healthcare teams by highlighting **who may need attention** —  
**all clinical decisions remain with humans**.

---

## 🔍 What This Project Does

- Analyzes IVR and SMS engagement patterns
- Estimates a **risk score (0–1)** for medication non-adherence
- Categorizes patients into **Low / Medium / High risk**
- Generates **clear explanations** for each prediction
- Provides access via:
  - A **Streamlit dashboard** for visual review
  - A **Flask REST API** for integration

---

## 🎯 Objective

The objective of this project is to demonstrate how **interpretable AI** can support early detection of medication adherence risks when real-world data is incomplete or limited.

The system focuses on:
- Transparency over black-box predictions
- Decision support rather than automation
- Practical usability in healthcare workflows

---

## 🌍 Impact & Applications

This project is suitable for:

- Public health monitoring programs  
- Chronic disease management initiatives  
- Remote patient engagement systems  
- Clinical workflow support tools  
- Academic and internship evaluations  

It is especially useful in environments where **manual follow-up resources are limited**.

---

## 🧠 System Overview (High-Level)

At a high level, the system:

1. Ingests patient engagement signals (IVR / SMS)
2. Computes a medication adherence risk score
3. Generates human-readable explanations
4. Displays results via dashboard or API

📐 **Detailed technical design is documented separately:**  
➡️ (Docs/model_architecture.md) 

---

## 📊 Model Performance & Evaluation

The model was evaluated using **synthetic patient data**, designed to reflect realistic medication adherence patterns.

### Evaluation Metrics Used

- **Precision** – How many flagged patients were truly at risk  
- **Recall** – How many at-risk patients were correctly identified  
- **F1 Score** – Balance between precision and recall  
- **ROC-AUC** – Overall ability to distinguish risk levels  

### Observed Performance (Synthetic Data)

| Metric       | Value |
|-------------|-------|
| Precision   | ~0.82 |
| Recall      | ~0.76 |
| F1 Score    | ~0.79 |
| ROC-AUC     | ~0.88 |

### Interpretation

- The system prioritizes **reducing false alerts**, helping avoid alert fatigue
- Risk thresholds can be adjusted based on organizational needs
- Performance is intended to **support early review**, not final diagnosis

Detailed outputs and visual evidence are available in the Results/folder.

---

## 🖥️ Interfaces

### Streamlit Dashboard

The Streamlit application provides:
- Risk distribution overview
- Individual patient risk assessment
- Visual summaries for reviewers

Run locally: streamlit run streamlit_app/app.py

### Flask REST API

The Flask API enables:
- Single and batch risk predictions
- Programmatic access to explanations
- Easy integration with external systems

Run locally: python -m src.api.flask_app

API available at: http://localhost:5000

---

## 📁 Repository Structure 

medication_buddy_system/
├── src/
│   ├── models/        # Risk scoring logic
│   ├── api/           # Flask API
│   └── train_model.py # Model training
│
├── streamlit_app/
│   └── app.py         # Dashboard UI
│
├── Docs/
│   ├── model_architecture.md
│   └── ethics_and_privacy.md
│
├── Results/
│   ├── plots/
│   └── reports/
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Data & References 

This project uses synthetic data only for development and testing.

### Reference resources: 
**WHO – Adherence to Long-Term Therapies**  : https://www.who.int/publications/i/item/WHO-MSD-03.01

**NIH – Medication Adherence Review**  : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6045499/

---

## 🔐 Ethics & Privacy

- No real patient data is stored
- No automated medical decisions are made
- All outputs require human review
- Designed with privacy-first principles

📘 Full details available in: ➡️ Docs/ethics_and_privacy.md

**⚠️ Disclaimer**: This is a research/demo system. Always consult healthcare professionals and conduct thorough validation before clinical deployment.
