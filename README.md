# Semiconductor-Fault-Detection-Dual-System
**Cost-Sensitive XGBoost + Explainable Bayesian Network for Semiconductor Fault Prediction (UCI SECOM). High recall + TCQ optimization + RCA.**

### Project Overview & Business Objective
This project develops a **dual-layer monitoring system** to predict and diagnose defective products in semiconductor manufacturing using the SECOM dataset (UCI Machine Learning Repository).  

**Core business problem**:  
In semiconductor production, the cost of missing a single defect (False Negative) can be **hundreds of times higher** than the cost of a false alarm (False Positive) — leading to scrapped wafers, rework, customer claims, and lost contracts. Defects are extremely rare (~6.6% Fail rate, 1:14 imbalance), yet their impact is catastrophic.  

The goal is to build a system that:  
- **Detects failures early** with high recall to minimize FN cost.  
- **Explains root causes** transparently (XAI) to guide fast corrective action.  
- **Optimizes total economic cost** (Total Cost of Quality – TCQ). 

### Dataset
- **Source**: SECOM (UCI ML Repository).  
- **Size**: 1567 samples, 590 anonymous sensor features.  
- **Target**: Pass/Fail (converted to 0 = Pass, 1 = Fail).  
- **Imbalance**: ~93.4% Pass, ~6.6% Fail (1:14 ratio).  

### Proposed Solution – Dual-Monitoring Architecture
A two-stage hybrid system combining predictive power and explainability:

**Stage 1 – Early Warning Station (Cost-Sensitive XGBoost)**  
- Hybrid Feature Selection: Boruta (all-relevant) + RFECV (recall-oriented) → reduced from 590 to **13 key features**  
- Hyperparameter tuning: Bayesian Optimization (Optuna)  
- Threshold optimization: Directly minimizes **Total Cost of Quality** (C_FN = 100, C_FP = 1)  
- Focus: High recall + low economic cost

**Stage 2 – Diagnosis Station (Explainable Bayesian Network)**  
- Transforms raw sensor data into **interpretable states** using k-means discretization (Elbow-optimized).  
- Learns causal structure via Hill-Climbing (BIC) → captures dependencies between process variables.  
- Estimates probabilistic relationships through Conditional Probability Tables (CPTs) using Maximum Likelihood Estimation (MLE).  
- Performs real-time inference (Variable Elimination) to identify **likely root causes** when a failure alert is triggered.  
- Provides **actionable insights**: key contributing sensors, risk uplift, and interpretable failure pathways.

### Key Metrics & Prioritization
- Primary: **Recall (Fail)**, **Geometric Mean (GM)**, **Total Cost of Quality (TCQ)**.  
- Secondary: Precision, PR-AUC, F1 (for reference only).  
- Accuracy and ROC-AUC are deprioritized due to severe imbalance.

### Main Results & Business Value
- High recall on minority class (Fail) with optimized economic cost.  
- Transparent root-cause diagnosis: identifies which sensors and states most increase failure risk.  
- Demonstrates clear economic advantage over traditional models (lower TCQ, faster root-cause resolution).

### Technologies
- Python 3.10+  
- XGBoost, Optuna, pgmpy, scikit-learn, imbalanced-learn  
- Visualization: Matplotlib, Seaborn, Plotly, NetworkX  

**Business Impact**  
- Reduces cost of missed defects through early, high-recall detection.  
- Shortens diagnostic time via automated root-cause insights.  
- Enables condition-based maintenance instead of scheduled checks.  
- Ready for real-world pilot deployment in semiconductor fabs.

**Copyright**  
© 2025 TÚ. All Rights Reserved.  
This project is for personal portfolio and demonstration purposes only. Unauthorized copying, modification, distribution, or use is strictly prohibited without prior written permission.

### Table of Contents
- [1. Import Libraries & Setup](#1-import-libraries--setup)
- [2. Load & Merge SECOM Dataset](#2-load--merge-secom-dataset)
- [3. Exploratory Data Analysis & Preprocessing](#3-exploratory-data-analysis--preprocessing)  
  - [3.1 Data Nature: Time-Series or Cross-Sectional?](#31-data-nature-time-series-or-cross-sectional)  
  - [3.2 Standardizing Target Variable](#32-standardizing-target-variable)  
  - [3.3 Removing Constant Features](#33-removing-constant-features)  
  - [3.4 Missing Values Analysis](#34-missing-values-analysis)  
  - [3.5 KNN-Based Imputation](#35-knn-based-imputation)  
  - [3.6 Class Imbalance Visualization](#36-class-imbalance-visualization)  
  - [3.7 Outlier Analysis](#37-outlier-analysis)  
  - [3.8 Correlation Analysis](#38-correlation-analysis)  

- [4. Metrics Definition & Prioritization](#4-metrics-definition--prioritization)  
- [5. Hybrid Feature Selection](#5-hybrid-feature-selection)  
- [6. Stage 1: Early Warning Station](#6-stage-1-early-warning-station)  
- [7. Stage 2: Diagnosis Station](#7-stage-2-diagnosis-station)  
  - [7.1 Discretisation for Bayesian Network](#71-discretisation-for-bayesian-network)  
  - [7.2 Causal Structure Learning](#72-causal-structure-learning)  
  - [7.3 Parameter Learning & Inference](#73-parameter-learning--inference)    
- [8. Web App Demo](#8-web-app-demo)  

**Next steps**:  
- Add interactive Streamlit demo  
- Deploy dashboard  
- Future work: real-time streaming, online learning
