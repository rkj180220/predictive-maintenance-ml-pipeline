# Quick Start Guide - ML Pipeline

## For macOS Users (IMPORTANT)

### Step 1: Install OpenMP (Required for XGBoost)
```bash
brew install libomp
```

### Step 2: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 3: Verify Installation
```bash
python -m src.ml_pipeline.model_preparation
```

You should see the model selection report displayed.

### Step 4: Run Complete ML Pipeline
```bash
python run_ml_pipeline.py
```

## Expected Runtime

- **Data Pipeline (if needed):** 5-10 minutes
- **ML Pipeline Total:** 5-15 minutes
  - Model Preparation: < 1 second
  - Model Training: 2-8 minutes (depends on CPU)
  - Model Evaluation: 10-30 seconds
  - MLOps Monitoring: 5-10 seconds

## Output Files Generated

After running `python run_ml_pipeline.py`:

1. **Models/** (2 files)
   - `random_forest.pkl`
   - `xgboost.pkl`

2. **Logs/** (8+ files)
   - `ml_pipeline_YYYYMMDD_HHMMSS.log`
   - `training_history.json`
   - `evaluation_results.json`
   - `evaluation_report.txt`
   - `mlops_metrics_random_forest_*.json`
   - `mlops_metrics_xgboost_*.json`
   - `mlops_all_metrics_*.json`
   - `data_drift_*.json`

3. **Visualizations/** (2 files)
   - `confusion_matrices.png`
   - `metrics_comparison.png`

## Troubleshooting

### Issue: "Processed data not found"
**Solution:** Run data pipeline first:
```bash
python main.py --mode once
```

### Issue: "XGBoost Library could not be loaded"
**Solution:** Install OpenMP:
```bash
brew install libomp
```

### Issue: "MemoryError"
**Solution:** Close other applications or reduce dataset size in code

## Success Indicators

You'll know the pipeline succeeded when you see:

```
✅ SUCCESS! ML Pipeline completed successfully
📊 Best Model: [Random Forest or XGBoost]
📁 Models saved to: /path/to/models
📁 Logs saved to: /path/to/logs
📁 Visualizations saved to: /path/to/visualizations
```

## Next Steps

1. Review evaluation results in `logs/evaluation_report.txt`
2. Check model comparison visualization in `visualizations/metrics_comparison.png`
3. Review MLOps metrics in `logs/mlops_all_metrics_*.json`
4. Use trained models for predictions (see API documentation)

---

**Questions?** Refer to `ML_PIPELINE_README.md` for detailed documentation.

