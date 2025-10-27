# Quick Reference Guide - Dashboard & API

## 🚀 START HERE

### One Command to Start Everything
```bash
cd /Users/ramkumarjayakumar/Dev/predictive-maintenance-ml-pipeline
python -m src.api.main
```

### Three URLs You Need

| Purpose | URL |
|---------|-----|
| **Dashboard** | http://127.0.0.1:8000/dashboard |
| **API Docs** | http://127.0.0.1:8000/docs |
| **Root** | http://127.0.0.1:8000 |

---

## 📊 Dashboard Quick Navigation

### Overview Page (Landing Page)
**What**: System health, model metrics, resources
**How to access**: Default page on load
**Key metrics**: Best model, accuracy, CPU/Memory/Disk

### Data Pipeline Page
**What**: Correlation, drift, validation, stats
**How to access**: Click "Data Pipeline" in sidebar
**Data from**: logs/correlation_analysis.json, logs/data_drift_*.json

### ML Pipeline Page
**What**: Model evaluation, features, health, metrics
**How to access**: Click "ML Pipeline" in sidebar
**Data from**: logs/evaluation_results.json, logs/feature_importance.json

### Monitoring Page
**What**: Real-time CPU, Memory, Disk usage
**How to access**: Click "Monitoring" in sidebar
**Features**: Auto-refresh, adjustable intervals

### Control Panel
**What**: Trigger pipelines, view logs
**How to access**: Click "Control Panel" in sidebar
**Actions**: Run data/ML pipelines, view execution status

### Logs Page
**What**: Filter and view execution logs
**How to access**: Click "Logs" in sidebar
**Features**: Search, color-coded levels

---

## 🔌 API Quick Reference

### Health Check
```bash
curl http://127.0.0.1:8000/api/v1/health
```

### Get All Dashboard Data
```bash
curl http://127.0.0.1:8000/api/v1/dashboard/overview
```

### Data Pipeline - Correlations
```bash
curl http://127.0.0.1:8000/api/v1/data/correlation-analysis
```

### ML Pipeline - Model Metrics
```bash
curl http://127.0.0.1:8000/api/v1/ml/evaluation-results
```

### ML Pipeline - XGBoost Metrics Only
```bash
curl "http://127.0.0.1:8000/api/v1/ml/mlops-metrics?model=xgboost"
```

### Trigger Data Pipeline
```bash
curl -X POST http://127.0.0.1:8000/api/v1/dashboard/trigger/data-pipeline
```

### Get Last 5 Pipeline Logs
```bash
curl "http://127.0.0.1:8000/api/v1/ml/pipeline-logs?limit=5"
```

---

## 📁 Key Files Location

| File | Path | Purpose |
|------|------|---------|
| Dashboard HTML | `src/api/templates/dashboard.html` | Main UI (850+ lines) |
| Dashboard CSS | `src/api/static/css/dashboard.css` | Styling (1100+ lines) |
| Dashboard JS | `src/api/static/js/dashboard.js` | Logic (900+ lines) |
| Dashboard Service | `src/api/services/dashboard_service.py` | Backend logic |
| Dashboard Routes | `src/api/routes/dashboard_routes.py` | API endpoints |
| Data Service | `src/api/services/data_pipeline_service.py` | Data pipeline logic |
| ML Service | `src/api/services/ml_pipeline_service.py` | ML pipeline logic |
| Main App | `src/api/main.py` | FastAPI server |

---

## 🎯 Common Tasks

### Task 1: Check System Health
1. Open dashboard
2. Look at Overview page (loads automatically)
3. See "System Health Status" card
4. Green ✓ = Healthy, Red ✗ = Error

### Task 2: View Model Performance
1. Click "ML Pipeline" in sidebar
2. Click "Load Results" button
3. See XGBoost and Random Forest metrics
4. Compare accuracy, precision, recall, F1

### Task 3: Check Data Quality
1. Click "Data Pipeline" in sidebar
2. Click "Load Data" under Statistical Analysis
3. See distribution stats for all features
4. Check for anomalies

### Task 4: Run Data Pipeline
1. Click "Control Panel" in sidebar
2. Click "Trigger Execution" under Data Pipeline
3. See execution status
4. Check logs in "Logs" page

### Task 5: Monitor in Real-time
1. Click "Monitoring" in sidebar
2. Enable "Auto-refresh" checkbox
3. Set refresh interval (e.g., 30 seconds)
4. Click "Start" button
5. Watch CPU, Memory, Disk update live

### Task 6: Search Logs
1. Click "Logs" in sidebar
2. Enter search term in filter box
3. Click "Filter" button
4. See only matching log entries

---

## 📈 API Endpoint Summary

### By Category

**Application Details (7)**
- `/api/v1/health` - Health check
- `/api/v1/application-details` - All details
- `/api/v1/pipeline/info` - Pipeline info
- `/api/v1/models/info` - Model info
- `/api/v1/metrics` - Performance metrics
- `/api/v1/data-pipeline` - Data status
- `/api/v1/system` - System info

**Data Pipeline (8)**
- `/api/v1/data/correlation-analysis` - Correlations
- `/api/v1/data/statistical-analysis` - Statistics
- `/api/v1/data/validation-report` - Validation
- `/api/v1/data/drift-analysis` - Latest drift
- `/api/v1/data/drift-history` - All drift records
- `/api/v1/data/pipeline-metrics` - Execution metrics
- `/api/v1/data/quality-summary` - Quality summary
- `/api/v1/data/all-details` - All data info

**ML Pipeline (9)**
- `/api/v1/ml/evaluation-results` - Model metrics
- `/api/v1/ml/feature-importance` - Feature rankings
- `/api/v1/ml/mlops-metrics` - MLOps data (optional: ?model=)
- `/api/v1/ml/mlops-all-metrics` - Aggregated metrics
- `/api/v1/ml/dashboard-data` - Dashboard data
- `/api/v1/ml/model-comparison` - Model comparison
- `/api/v1/ml/pipeline-logs` - Logs (optional: ?limit=10)
- `/api/v1/ml/model-health` - Model health
- `/api/v1/ml/all-details` - All ML info

**Dashboard (4)**
- `GET /api/v1/dashboard/overview` - Dashboard overview
- `POST /api/v1/dashboard/trigger/data-pipeline` - Trigger data
- `POST /api/v1/dashboard/trigger/ml-pipeline` - Trigger ML
- `GET /api/v1/dashboard/system-stats` - System resources

---

## ⚙️ Configuration

### Change Port
Edit `src/api/main.py` (last line):
```python
uvicorn.run(
    "src.api.main:app",
    host="127.0.0.1",
    port=8000,  # Change here
    ...
)
```

### Change Primary Color
Edit `src/api/static/css/dashboard.css` (line 8):
```css
--primary-color: #2563eb;  /* Change this hex code */
```

### Adjust Monitoring Interval
In dashboard, go to Monitoring page:
- Default: 30 seconds
- Min: 5 seconds
- Max: 300 seconds

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Dashboard won't load | Check API is running, verify URL, open browser console (F12) |
| API returns error | Check log files in `logs/` directory, verify endpoint URL |
| Data not showing | Verify log files exist, check API response in Network tab |
| Slow performance | Reduce monitoring refresh interval, clear browser cache |
| Port 8000 in use | Change port in `src/api/main.py` or kill process using port |
| Module not found | Install dependencies: `pip install -r requirements.txt` |

---

## 📚 Full Documentation

For complete details, see:
- **`DASHBOARD_GUIDE.md`** - Complete dashboard documentation
- **`API_DOCUMENTATION.md`** - Full API reference
- **`API_QUICKSTART.md`** - Quick start with examples
- **`DASHBOARD_AND_API_COMPLETE.md`** - Complete implementation summary

---

## ✅ Verification Checklist

Before deployment, verify:
- [ ] API server starts without errors
- [ ] Dashboard loads at http://127.0.0.1:8000/dashboard
- [ ] Overview page shows data
- [ ] Data Pipeline page loads correlation analysis
- [ ] ML Pipeline page shows model metrics
- [ ] Monitoring page updates in real-time
- [ ] Control Panel can trigger pipelines
- [ ] Logs page displays execution logs
- [ ] Dark mode toggle works
- [ ] Dashboard is responsive on mobile

---

## 🎯 What Data Comes From Where

### Dashboard Gets Data From:

**Overview Page**
- `logs/statistical_analysis.json` → Records & features
- `logs/evaluation_results.json` → Best model & accuracy
- `logs/mlops_dashboard_data.json` → Model health
- System resources → Direct calculation

**Data Pipeline Page**
- `logs/correlation_analysis.json` → Correlations
- `logs/statistical_analysis.json` → Statistics
- `logs/data_drift_*.json` → Drift detection
- `logs/data_validation_report.txt` → Validation

**ML Pipeline Page**
- `logs/evaluation_results.json` → Evaluation metrics
- `logs/feature_importance.json` → Feature rankings
- `logs/mlops_metrics_*.json` → MLOps monitoring
- `logs/mlops_dashboard_data.json` → Health status

**Monitoring Page**
- Real-time system stats (CPU, Memory, Disk)

**Control Panel Page**
- `logs/ml_pipeline_*.log` → Pipeline logs
- Execute: `python run_ml_pipeline.py`

**Logs Page**
- Any log files in `logs/` directory

---

## 🚀 Deployment Commands

### Development
```bash
python -m src.api.main
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 src.api.main:app
```

### Docker
```bash
docker run -p 8000:8000 predictive-maintenance:latest
```

---

## 📞 Quick Help

**How to view API documentation?**
→ Open http://127.0.0.1:8000/docs (Swagger UI)

**How to trigger a pipeline?**
→ Control Panel page → Click "Trigger Execution"

**How to see all metrics?**
→ Overview page shows summary, ML Pipeline page shows details

**How to export data?**
→ API returns JSON, copy to file or use curl redirection

**How to customize dashboard?**
→ Edit CSS colors in `src/api/static/css/dashboard.css`

**How to add new page?**
→ Add HTML section, CSS styling, JS functions, navigation link

**How to add new API endpoint?**
→ Create service method, add route, add UI component

---

## 📊 System Overview

```
Browser
   ↓ (User navigates dashboard)
Dashboard UI (HTML/CSS/JS)
   ↓ (Fetch API calls)
FastAPI Routes (/api/v1/*)
   ↓ (Process request)
Services (Dashboard/Data/ML)
   ↓ (Read data)
Log Files (logs/)
   ↓ (Return JSON)
Dashboard UI
   ↓ (Display)
User sees data
```

---

## ✨ Features at a Glance

- ✅ **6 Dashboard Pages** - Overview, Data, ML, Monitor, Control, Logs
- ✅ **28 API Endpoints** - Comprehensive coverage
- ✅ **Real-time Monitoring** - Live system stats
- ✅ **Pipeline Control** - Trigger from UI
- ✅ **Data Integration** - All log files accessible
- ✅ **Responsive Design** - Works on all devices
- ✅ **Dark Mode** - Comfortable viewing
- ✅ **Professional UI** - Modern design
- ✅ **Error Handling** - User-friendly messages
- ✅ **Production Ready** - Fully functional

---

## 🎉 You're All Set!

Everything is ready. Just:

1. **Run**: `python -m src.api.main`
2. **Open**: http://127.0.0.1:8000/dashboard
3. **Explore**: Click through all pages
4. **Enjoy**: Professional ML monitoring dashboard

**Total Code Written: 3000+ lines**
**Endpoints Created: 28**
**Dashboard Pages: 6**
**Time to Deploy: < 1 minute**

🚀 **Happy monitoring!**

