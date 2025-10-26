# Machine Learning Pipeline Documentation

## Sub-Objective 2: Machine Learning Pipeline for Predictive Maintenance

**Author:** ramkumarjayakumar  
**Date:** October 26, 2025  
**Status:** ✅ Complete

---

## 📋 Overview

This document describes the complete Machine Learning Pipeline implementation for the Predictive Maintenance project. The pipeline implements all requirements from Sub-Objective 2 (Activities 2.1 - 2.4).

---

## 🎯 Activities Completed

### ✅ Activity 2.1: Model Preparation

**File:** `src/ml_pipeline/model_preparation.py`

**Selected Algorithms:**

1. **Random Forest Classifier** (Primary)
   - **Justification:**
     - Handles high-dimensional data (100+ features) effectively
     - Built-in feature importance for interpretability
     - Robust to outliers in sensor data
     - No feature scaling required
     - Handles class imbalance with `class_weight='balanced'`
   
   - **Hyperparameters:**
     ```python
     n_estimators=200
     max_depth=20
     min_samples_split=10
     min_samples_leaf=4
     max_features='sqrt'
     class_weight='balanced'
     ```

2. **XGBoost Classifier** (Secondary)
   - **Justification:**
     - State-of-the-art gradient boosting performance
     - Excellent with engineered time-series features
     - Built-in regularization prevents overfitting
     - Handles imbalanced data with `scale_pos_weight`
     - Industry-leading accuracy for failure prediction
   
   - **Hyperparameters:**
     ```python
     n_estimators=200
     max_depth=10
     learning_rate=0.1
     subsample=0.8
     colsample_bytree=0.8
     scale_pos_weight=6.3
     ```

**Why These Algorithms?**
- **Random Forest:** Reliable baseline, interpretable, handles imbalanced data
- **XGBoost:** Superior performance, proven in production environments
- Both are tree-based algorithms ideal for tabular sensor data
- Complementary strengths for comparison and ensemble potential

---

### ✅ Activity 2.2: Model Training

**File:** `src/ml_pipeline/model_training.py`

**Implementation Details:**

1. **Data Split:** 70% Training / 30% Testing
   - Stratified split maintains class distribution
   - Reproducible with `random_state=42`
   - Approximately 80,505 training samples
   - Approximately 34,503 test samples

2. **Training Process:**
   - Load processed data from `data/processed/train_processed.parquet`
   - Separate features and target (`failure_label`)
   - Handle missing/infinite values
   - Train both models sequentially
   - Extract feature importance
   - Save models to `models/` directory

3. **Training Metadata Tracked:**
   - Training time (seconds and minutes)
   - Number of samples and features
   - Top 10 most important features
   - Timestamp and configuration

---

### ✅ Activity 2.3: Model Evaluation

**File:** `src/ml_pipeline/model_evaluation.py`

**Evaluation Metrics (4+ Required):**

**Core Metrics:**
1. **Accuracy** - Overall prediction correctness
2. **Precision** - Positive prediction accuracy (minimize false alarms)
3. **Recall** - Failure detection rate (minimize missed failures)
4. **F1-Score** - Harmonic mean of precision and recall

**Additional Metrics:**
5. **ROC-AUC Score** - Area under ROC curve
6. **Balanced Accuracy** - Accounts for class imbalance
7. **Matthews Correlation Coefficient** - Overall quality measure
8. **Cohen's Kappa** - Agreement beyond chance

**Business Metrics:**
- **False Positive Rate** - Rate of unnecessary maintenance alerts
- **False Negative Rate** - Rate of missed failures (CRITICAL)
- **Specificity** - True negative rate
- **Cost Analysis** - Business impact calculation
  - False Negative Cost: $10,000 per missed failure
  - False Positive Cost: $1,000 per false alarm

**Outputs:**
- Confusion matrices for both models
- Side-by-side model comparison
- Visualizations (PNG format)
- JSON and text reports

---

### ✅ Activity 2.4: MLOps Monitoring

**File:** `src/ml_pipeline/mlops_monitoring.py`

**Monitoring Features:**

1. **Metric Logging (4+ metrics tracked):**
   - Accuracy, Precision, Recall, F1-Score
   - Extended metrics (ROC-AUC, MCC, Cohen's Kappa)
   - Error rates (FPR, FNR, TPR, Specificity)
   - Business impact metrics (costs, predictions)

2. **Model Health Assessment:**
   - Health score calculation (0-100)
   - Status: EXCELLENT / GOOD / FAIR / POOR
   - Automated issue detection
   - Actionable recommendations

3. **Data Drift Detection:**
   - Feature-level drift analysis
   - Statistical comparison (train vs test)
   - Drift score calculation (>2 std = drift)
   - Alerts for significant drift

4. **AWS CloudWatch Integration:**
   - Optional CloudWatch logging
   - Structured JSON logging
   - Real-time metric streaming
   - Production-ready monitoring

5. **Dashboard Data Generation:**
   - JSON output for visualization tools
   - Model comparison data
   - Historical metrics tracking

---

## 🚀 Running the ML Pipeline

### Quick Start (Recommended)

```bash
# Ensure data pipeline has been run first
python main.py --mode once

# Run complete ML pipeline
python run_ml_pipeline.py
```

### Programmatic Access

```python
from pathlib import Path
from src.ml_pipeline.pipeline_orchestrator import MLPipelineOrchestrator

# Initialize orchestrator
orchestrator = MLPipelineOrchestrator(Path.cwd())

# Run pipeline
results = orchestrator.run_complete_pipeline(enable_cloudwatch=False)

# Access results
print(f"Best Model: {results['model_evaluation']['best_model']}")
print(f"Accuracy: {results['model_evaluation']['random_forest']['accuracy']}")
```

### Step-by-Step Execution

```bash
# 1. Model Preparation
python -m src.ml_pipeline.model_preparation

# 2. Complete Pipeline (Training + Evaluation + MLOps)
python -m src.ml_pipeline.pipeline_orchestrator
```

---

## 📊 Expected Output

### Console Output Example

```
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
STARTING COMPLETE MACHINE LEARNING PIPELINE
Sub-Objective 2: ML Pipeline for Predictive Maintenance
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

================================================================================
STAGE 1: MODEL PREPARATION (Activity 2.1)
================================================================================
...
✅ Stage 1 Complete: Models prepared and initialized

================================================================================
STAGE 2: MODEL TRAINING (Activity 2.2)
================================================================================
...
✅ Stage 2 Complete: Models trained successfully

================================================================================
STAGE 3: MODEL EVALUATION (Activity 2.3)
================================================================================
...
✅ Stage 3 Complete: Models evaluated successfully

================================================================================
STAGE 4: MLOPS MONITORING (Activity 2.4)
================================================================================
...
✅ Stage 4 Complete: MLOps monitoring configured

🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
```

### Generated Files

**Models:**
- `models/random_forest.pkl` (~50-100 MB)
- `models/xgboost.pkl` (~20-50 MB)

**Logs:**
- `logs/ml_pipeline_20251026_*.log`
- `logs/training_history.json`
- `logs/evaluation_results.json`
- `logs/evaluation_report.txt`
- `logs/mlops_metrics_random_forest_*.json`
- `logs/mlops_metrics_xgboost_*.json`
- `logs/mlops_all_metrics_*.json`
- `logs/data_drift_*.json`
- `logs/mlops_dashboard_data.json`

**Visualizations:**
- `visualizations/confusion_matrices.png`
- `visualizations/metrics_comparison.png`

---

## 📈 Performance Metrics (Expected)

Based on the NASA C-MAPSS dataset characteristics:

### Random Forest (Expected Performance)
- **Accuracy:** 92-96%
- **Precision:** 85-90%
- **Recall:** 80-88%
- **F1-Score:** 83-89%
- **ROC-AUC:** 0.90-0.95

### XGBoost (Expected Performance)
- **Accuracy:** 93-97%
- **Precision:** 87-92%
- **Recall:** 82-90%
- **F1-Score:** 85-91%
- **ROC-AUC:** 0.92-0.96

**Note:** Actual performance depends on feature engineering quality and hyperparameter tuning.

---

## 🔧 Troubleshooting

### Common Issues

1. **FileNotFoundError: Processed data not found**
   - **Solution:** Run data pipeline first: `python main.py --mode once`

2. **MemoryError during training**
   - **Solution:** Reduce dataset size or use sampling
   - Or increase available RAM

3. **ImportError: No module named 'xgboost'**
   - **Solution:** `pip install xgboost`

4. **Slow training time**
   - **Expected:** 2-5 minutes for Random Forest, 3-8 minutes for XGBoost
   - **Solution:** Reduce `n_estimators` for faster training

---

## 🎓 Key Learnings

1. **Model Selection:** Tree-based models excel with sensor data
2. **Class Imbalance:** Use `class_weight` and `scale_pos_weight`
3. **Feature Engineering:** Critical for predictive maintenance
4. **Business Metrics:** FN cost >> FP cost in maintenance scenarios
5. **MLOps:** Continuous monitoring essential for production

---

## 📚 References

- NASA C-MAPSS Dataset: [Kaggle Link](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- Random Forest: Scikit-learn Documentation
- XGBoost: [Official Documentation](https://xgboost.readthedocs.io/)
- MLOps Best Practices: Google Cloud MLOps Guide

---

## ✅ Completion Checklist

- [x] Activity 2.1: Model Preparation ✅
- [x] Activity 2.2: Model Training (70/30 split) ✅
- [x] Activity 2.3: Model Evaluation (4+ metrics) ✅
- [x] Activity 2.4: MLOps Monitoring ✅
- [x] Documentation Complete ✅
- [x] Code Tested ✅
- [x] README Updated ✅

**Status: SUB-OBJECTIVE 2 COMPLETE** 🎉

---

*Last Updated: October 26, 2025*

