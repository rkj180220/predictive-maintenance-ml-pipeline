# Predictive Maintenance ML Pipeline

## 🎯 Project Overview

A production-ready data pipeline for predictive maintenance of turbofan engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. This comprehensive system integrates data ingestion, preprocessing, feature engineering, exploratory data analysis, and automated monitoring with AWS integration capabilities.

**Author:** rkj180220  
**Date:** October 18, 2025  
**Dataset:** NASA C-MAPSS FD002 + FD004 (114,000+ records)

---

## 🚀 Features

### ✅ Comprehensive Pipeline Components

1. **Business Understanding**
   - Detailed problem definition and KPI framework
   - Cost-benefit analysis with ROI projections
   - Stakeholder impact assessment

2. **Data Ingestion**
   - Automated Kaggle dataset download
   - Multi-dataset handling (FD002 + FD004)
   - Data validation and quality checks
   - Support for 21 sensor measurements per engine cycle

3. **Data Preprocessing**
   - Missing value imputation strategies
   - Outlier detection and handling (IQR/Z-score methods)
   - Data normalization (Standard/MinMax/Robust scaling)
   - Low variance feature removal

4. **Feature Engineering**
   - Rolling window features (5, 10, 20, 50 cycles)
   - Lag features (1, 5, 10, 20 cycles)
   - Statistical aggregations (mean, std, min, max, median)
   - Trend indicators and rate of change
   - Interaction features between sensors
   - Time-based features (cycle normalization, life stage)

5. **Exploratory Data Analysis (EDA)**
   - Comprehensive statistical analysis
   - Correlation analysis for all 21 sensors
   - Interactive and static visualizations
   - Distribution analysis
   - Engine degradation pattern visualization
   - RUL (Remaining Useful Life) analysis

6. **Feature Importance**
   - Random Forest importance
   - Gradient Boosting importance
   - Correlation-based importance
   - Combined ranking methodology

7. **DataOps & Monitoring**
   - Scheduled execution every 2 minutes
   - Comprehensive logging (console + file + JSON)
   - AWS CloudWatch integration
   - Real-time monitoring dashboard
   - Performance metrics tracking
   - Alert system

---

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM recommended
- 2GB+ free disk space

### Python Dependencies
See `requirements.txt` for complete list. Key packages:
- pandas, numpy, scipy
- scikit-learn, xgboost, lightgbm
- matplotlib, seaborn, plotly
- fastapi, uvicorn
- boto3 (AWS integration)
- kaggle (dataset download)
- schedule, APScheduler

---

## 🛠️ Installation

### 1. Clone Repository
```bash
cd /path/to/your/workspace
# Or use existing directory
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API (Required for Dataset Download)

Create Kaggle API credentials:
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. Set permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 5. (Optional) Configure AWS Credentials

For AWS CloudWatch integration, create `.env` file:
```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=predictive-maintenance-data
AWS_CLOUDWATCH_LOG_GROUP=/aws/predictive-maintenance/pipeline
```

---

## 🎮 Usage

### One-Time Execution
Run the complete pipeline once:
```bash
python main.py --mode once
```

### Scheduled Execution (Every 2 Minutes)
Run the pipeline continuously with 2-minute intervals:
```bash
python main.py --mode scheduled
```

### Run Specific Components

**Business Understanding Only:**
```bash
python -m src.business_understanding.problem_definition
```

**Data Ingestion Only:**
```bash
python -m src.data_ingestion.kaggle_downloader
```

**EDA Only (requires processed data):**
```bash
python -m src.eda.statistical_analysis
```

---

## 📁 Project Structure

```
predictive-maintenance-ml-pipeline/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── .env                            # Environment variables (create this)
│
├── src/                            # Source code
│   ├── config/
│   │   ├── settings.py            # Configuration settings
│   │   └── aws_config.py          # AWS configuration
│   │
│   ├── business_understanding/
│   │   └── problem_definition.py  # Business problem definition
│   │
│   ├── data_ingestion/
│   │   ├── kaggle_downloader.py   # Kaggle dataset downloader
│   │   ├── data_validator.py      # Data validation
│   │   └── data_loader.py         # Data loading and merging
│   │
│   ├── preprocessing/
│   │   ├── data_cleaner.py        # Data cleaning
│   │   ├── feature_engineer.py    # Feature engineering
│   │   └── normalizer.py          # Data normalization
│   │
│   ├── eda/
│   │   ├── statistical_analysis.py # Statistical analysis
│   │   ├── correlation_analysis.py # Correlation analysis
│   │   ├── visualization.py        # Data visualization
│   │   └── feature_importance.py   # Feature importance
│   │
│   ├── dataops/
│   │   ├── logger.py              # Advanced logging
│   │   ├── monitoring.py          # Pipeline monitoring
│   │   ├── pipeline_scheduler.py  # Scheduling
│   │   └── dashboard.py           # Dashboard generation
│   │
│   └── pipeline_main.py           # Pipeline orchestrator
│
├── data/                          # Data directory
│   ├── raw/                       # Raw downloaded data
│   ├── processed/                 # Processed data
│   └── features/                  # Engineered features
│
├── logs/                          # Log files
├── visualizations/                # Generated plots
├── dashboards/                    # HTML dashboards
├── docs/                          # Documentation
└── notebooks/                     # Jupyter notebooks (optional)
```

---

## 📊 Output Files

After running the pipeline, you'll find:

### Data Files
- `data/raw/`: Original NASA C-MAPSS files
- `data/processed/train_processed.csv`: Processed training data
- `data/processed/test_processed.csv`: Processed test data
- `data/processed/train_processed.parquet`: Efficient storage format
- `data/processed/scalers.pkl`: Fitted normalizers

### Reports & Logs
- `logs/predictive_maintenance_YYYYMMDD.log`: Execution logs
- `logs/data_validation_report.txt`: Data quality report
- `logs/statistical_analysis.txt`: Statistical analysis report
- `logs/correlation_analysis.txt`: Correlation analysis report
- `logs/feature_importance.txt`: Feature importance report
- `logs/pipeline_metrics.json`: Execution metrics

### Visualizations
- `visualizations/correlation_heatmap.png`: Sensor correlation heatmap
- `visualizations/rul_distribution.png`: RUL distribution plots
- `visualizations/sensor_distributions.png`: Individual sensor distributions
- `visualizations/degradation_patterns.png`: Engine degradation over time
- `visualizations/rul_vs_sensors.png`: RUL vs sensor scatter plots
- `visualizations/feature_rul_correlation.png`: Feature-RUL correlations

### Dashboards
- `dashboards/dashboard.html`: Real-time monitoring dashboard

### Documentation
- `docs/business_problem_definition.txt`: Business context
- `docs/business_problem_definition.json`: Structured business info

---

## 🔍 Dataset Information

### NASA C-MAPSS Dataset
- **Source:** Kaggle (behrad3d/nasa-cmaps)
- **Datasets Used:** FD002 and FD004
- **Total Records:** 114,000+ sensor readings
- **Engines:** 509 turbofan engines
- **Sensors:** 21 sensor measurements per cycle
- **Operational Settings:** 3 operational conditions

### Data Description
- **unit_id:** Engine identifier
- **time_cycles:** Operational cycle number
- **setting_1, setting_2, setting_3:** Operational settings
- **sensor_1 to sensor_21:** Sensor measurements
- **RUL:** Remaining Useful Life (target variable)

---

## 📈 Key Performance Indicators (KPIs)

### Business KPIs
- **Cost Reduction Target:** 30% maintenance cost savings
- **Downtime Reduction:** 40% reduction in unplanned downtime
- **ROI Timeline:** 4-6 months payback period
- **5-Year ROI:** 4,250%

### Technical KPIs
- **Model Accuracy Target:** ≥ 90%
- **Precision Target:** ≥ 85%
- **Recall Target:** ≥ 90%
- **F1 Score Target:** ≥ 87%
- **False Negative Rate:** ≤ 5%
- **Data Quality Threshold:** ≥ 95%

---

## 🔧 Configuration

Key configuration parameters in `src/config/settings.py`:

```python
# Feature Engineering
FEATURE_CONFIG = {
    "rolling_windows": [5, 10, 20, 50],
    "lag_features": [1, 5, 10, 20],
    "statistical_features": ["mean", "std", "min", "max", "median"],
    "trend_window": 10
}

# DataOps
DATAOPS_CONFIG = {
    "pipeline_schedule_minutes": 2,  # Run every 2 minutes
    "log_level": "INFO",
    "max_log_files": 10,
    "log_rotation_mb": 10
}

# Model Configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "failure_threshold": 30  # RUL threshold for failure classification
}
```

---

## 🐛 Troubleshooting

### Kaggle API Issues
**Problem:** "Could not find kaggle.json"
```bash
# Solution: Set up Kaggle credentials
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Memory Issues
**Problem:** "MemoryError during feature engineering"
```python
# Solution: Reduce sample size in feature_engineer.py
# Or increase system RAM
```

### Missing Dependencies
**Problem:** "ModuleNotFoundError"
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --upgrade
```

### AWS Connection Issues
**Problem:** "Failed to initialize CloudWatch client"
```bash
# Solution: AWS integration is optional
# Pipeline will work without AWS credentials
# Just skip CloudWatch features
```

---

## 📝 Logging Levels

The pipeline uses hierarchical logging:
- **DEBUG:** Detailed diagnostic information
- **INFO:** General informational messages (default)
- **WARNING:** Warning messages for potential issues
- **ERROR:** Error messages for failures
- **CRITICAL:** Critical failures

Change log level in `src/config/settings.py`:
```python
DATAOPS_CONFIG = {
    "log_level": "DEBUG",  # Change to DEBUG for verbose logging
}
```

---

## 🚀 Next Steps & Extensions

### Planned Features
1. **Machine Learning Models**
   - RUL prediction models (Random Forest, XGBoost, LSTM)
   - Classification models for failure prediction
   - Model versioning and A/B testing

2. **API Development**
   - FastAPI REST endpoints
   - Real-time prediction API
   - Model serving infrastructure

3. **Advanced Monitoring**
   - Grafana dashboards
   - Prometheus metrics
   - Data drift detection

4. **Cloud Deployment**
   - Docker containerization
   - Kubernetes orchestration
   - CI/CD pipeline

---

## 📚 References

- NASA C-MAPSS Dataset: [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- Prognostics Center of Excellence: [NASA](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

## 👨‍💻 Author

**rkj180220**  
Data Engineer specializing in ML pipelines and predictive analytics

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 rkj180220

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**For educational and research purposes.**

---

## 📞 Contact

**Author:** rkj180220

- **GitHub:** [@rkj180220](https://github.com/rkj180220)
- **Email:** [Your email address]
- **LinkedIn:** [Your LinkedIn profile]

**Project Repository:** https://github.com/rkj180220/predictive-maintenance-ml-pipeline

For questions, suggestions, or collaboration opportunities, feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact via email

---

## 🤝 Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests** for your changes and ensure all existing tests pass

3. **Follow code standards:**
   - Use PEP 8 style guide for Python code
   - Add docstrings to all functions and classes
   - Keep functions focused and modular
   - Add type hints where applicable

4. **Run linters and tests locally** before submitting:
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run linters
   flake8 src/
   black src/ --check
   mypy src/
   ```

5. **Open a pull request** with:
   - Clear description of changes
   - Related issue numbers
   - Screenshots/logs if applicable

6. **Update documentation** if adding new features

Include a `CHANGELOG.md` entry for non-trivial changes.

---

## 🧪 Tests

The project uses `pytest` for unit and integration testing.

### Running Tests

**Run all tests:**
```bash
pytest tests/ -v
```

**Run tests with coverage:**
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

**Run specific test file:**
```bash
pytest tests/test_data_cleaner.py -v
```

**Run specific test function:**
```bash
pytest tests/test_feature_engineer.py::test_rolling_features -v
```

### Test Structure
```
tests/
├── test_data_ingestion.py      # Data ingestion tests
├── test_preprocessing.py        # Preprocessing tests
├── test_feature_engineering.py  # Feature engineering tests
├── test_eda.py                 # EDA tests
└── test_pipeline.py            # End-to-end pipeline tests
```

### Coverage Goals
- Minimum coverage target: 80%
- Critical modules (data_cleaner, feature_engineer): 90%+

---

## 🔨 Pre-commit Hooks

Install pre-commit hooks to automatically check code quality before committing:

### Setup
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Configured Hooks
The `.pre-commit-config.yaml` includes:
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML files
- **black**: Auto-format Python code
- **flake8**: Lint Python code
- **isort**: Sort imports

---

## 🔄 Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/ci.yml` for automated testing:

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8 black
    
    - name: Run linters
      run: |
        flake8 src/ --max-line-length=100
        black src/ --check
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### CI Pipeline Features
- ✅ Multi-version Python testing (3.8, 3.9, 3.10)
- ✅ Code quality checks (flake8, black)
- ✅ Automated test execution
- ✅ Coverage reporting
- ✅ Dependency caching for faster builds

---

## 🐳 Docker

Build and run the pipeline in a containerized environment.

### Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/features logs visualizations dashboards

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the pipeline
CMD ["python", "main.py", "--mode", "once"]
```

### Docker Commands

**Build the image:**
```bash
docker build -t predictive-maintenance:latest .
```

**Run once:**
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/visualizations:/app/visualizations \
  predictive-maintenance:latest
```

**Run in scheduled mode:**
```bash
docker run -d \
  --name pm-pipeline \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  predictive-maintenance:latest \
  python main.py --mode scheduled
```

**With environment variables:**
```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  predictive-maintenance:latest
```

### Docker Compose

Create `docker-compose.yml` for easier orchestration:

```yaml
version: '3.8'

services:
  pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./visualizations:/app/visualizations
      - ./dashboards:/app/dashboards
    env_file:
      - .env
    command: python main.py --mode scheduled
    restart: unless-stopped
```

**Run with Docker Compose:**
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## 🌐 Running the API (FastAPI)

If implementing a prediction API, use FastAPI for serving models.

### Start the API Server

```bash
# Development mode with auto-reload
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints (Planned)

- `GET /` - API health check
- `GET /api/v1/health` - Detailed health status
- `POST /api/v1/predict` - Predict RUL for engine data
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/models` - List available models
- `GET /api/v1/metrics` - Pipeline metrics

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict RUL
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": 1,
    "time_cycles": 150,
    "setting_1": 0.0023,
    "sensor_1": 518.67,
    "sensor_2": 642.45
  }'
```

### API Documentation
Once running, access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🏷️ Release & Versioning

The project follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **MAJOR:** Incompatible API changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes (backward compatible)

### Creating a Release

```bash
# Update version in setup.py or __version__.py
# Update CHANGELOG.md

# Commit changes
git add .
git commit -m "Release v1.0.0"

# Create and push tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Push to main branch
git push origin main
```

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number bumped
- [ ] Tag created
- [ ] Release notes prepared

---

## 🔐 Security

### Best Practices

1. **Never commit secrets:**
   - Use `.env` files (add to `.gitignore`)
   - Use environment variables
   - Use secrets management tools (AWS Secrets Manager, HashiCorp Vault)

2. **Kaggle Credentials:**
   ```bash
   # Store in user directory
   ~/.kaggle/kaggle.json
   # Set proper permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **AWS Credentials:**
   ```bash
   # Use environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   
   # Or use AWS CLI configuration
   aws configure
   ```

4. **Rotate credentials periodically:**
   - AWS keys: Every 90 days
   - API tokens: Every 6 months

5. **Code scanning:**
   - Add Dependabot for dependency updates
   - Use `safety` to check for known vulnerabilities:
   ```bash
   pip install safety
   safety check
   ```

6. **Secret scanning:**
   - Add git-secrets or similar tools
   - Enable GitHub secret scanning
   - Use pre-commit hooks for secret detection

### Security Reporting
Report security vulnerabilities privately to: [Your contact email]

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 rkj180220

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**For educational and research purposes.**

---

## 📞 Contact

**Author:** rkj180220

- **GitHub:** [@rkj180220](https://github.com/rkj180220)
- **Email:** [Your email address]
- **LinkedIn:** [Your LinkedIn profile]

**Project Repository:** https://github.com/rkj180220/predictive-maintenance-ml-pipeline

For questions, suggestions, or collaboration opportunities, feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact via email

---

## 🙏 Acknowledgments

- NASA Ames Research Center for the C-MAPSS dataset
- Kaggle community for dataset hosting
- Open-source contributors of all dependencies

---

**Last Updated:** October 18, 2025
