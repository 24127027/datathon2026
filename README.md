# Datathon 2026 - Revenue & COGS Forecasting

A comprehensive Machine Learning pipeline designed to solve complex financial forecasting challenges in e-commerce. The primary goal of this project is to accurately predict two critical metrics: **Revenue** and **COGS (Cost of Goods Sold)**.

This project focuses heavily on addressing real-world time-series anomalies such as:
- **Structural Regime Shifts:** Handling sudden drops in sales volume while web traffic continued to grow (specifically addressing the structural shock starting in 2019).
- **Cyclicality & Anomalies:** Capturing biennial cycles (2-year patterns), end-of-month revenue spikes, and intermittent shopping peaks.

---

## 📂 Repository Structure

The project is structured following standard Data Science practices as a Python package:

```text
datathon2026/
├── data/                       # Raw and processed datasets
│   └── datathon-2026-round-1/  # Subdirectories: master, transaction, operational, analytical
├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│   ├── eda.ipynb               # Exploratory Data Analysis
│   ├── base.ipynb              # Baseline modeling and pipelines
│   ├── submission.ipynb        # Final submission generation notebook
│   ├── profit.ipynb            # Profit margin analysis
│   ├── image.ipynb             # Data visualization
│   ├── operation_exploration.ipynb
│   └── time_feature.ipynb      # Time-based feature experimentation
├── reports/                    # Generated reports and evaluation results
├── src/                        # Core Python package
│   ├── data/                   # Data loaders and validation (loader.py, validate.py)
│   ├── evaluation/             # Evaluation metrics: MAE, RMSE, MAPE, sMAPE, R2 (metrics.py)
│   ├── features/               # Feature engineering logic (build.py, select.py)
│   ├── models/                 # ML model wrappers (sklearn_models.py)
│   ├── pipelines/              # Train/Infer workflows (train.py, infer.py)
│   └── utils/                  # Helper utilities (paths, logging)
├── pyproject.toml              # Project metadata and dependencies (PEP 621)
└── uv.lock                     # Lock file for strict dependency management
```

---

## 🛠 Core Workflow and Features

### 1. Data Loading
The project processes data from multiple complex, relational sources simulating a real-world ERP/CRM system (`src/data/loader.py`):
- **Master Data:** Customers, Products, Geography, Promotions.
- **Transaction Data:** Orders, Order Items, Payments, Shipments, Returns, Reviews.
- **Operational Data:** Inventory, Web Traffic.
- **Analytical Data:** Historical Sales containing the targets (`Revenue`, `COGS`).

### 2. Feature Engineering
The feature creation process (`src/features/build.py`) is the core of the solution, capable of generating over 100 critical features:
- **Time & Calendar Features:** Day, month, year, and cyclical transformations (sin/cos) to capture seasonality.
- **Lags & Rolling Windows:** Auto-generates historical features (shift 1, 7, 14, 28, 365, 730 days) and rolling means/standard deviations.
- **Business Metrics:** Gross Margin, AOV (Average Order Value), Conversion Rate, Bounce Rate, Return Rate, and Inventory Pressure.
- **Regime & Context Features:**
  - `shock_period`: Flags the period from 2019-03 to 2022-12 where the structural break in revenue occurred.
  - `is_month8_odd`: Captures a specific 2-year cycle anomaly occurring in August.
  - `is_peak_period`: Identifies start and end-of-month shopping peaks.

### 3. Modeling
The project uses a unified configuration file `SklearnRegressorConfig` (`src/models/sklearn_models.py`), supporting powerful tree-based algorithms:
- **LightGBM** (Currently the best-performing configuration used in the submission pipeline)
- **XGBoost**
- **CatBoost**
- **Random Forest** & **Gradient Boosting**

All models support built-in visual explanations via Feature Importance and **SHAP** dependency plots.

### 4. Evaluation Metrics
Located in `src/evaluation/metrics.py`, the project measures performance using standard forecasting metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE / sMAPE (Mean Absolute Percentage Error / Symmetric MAPE)
- R-squared (R2 Score)
Includes support for calculating metrics across specific segments (Grouped Metrics) to identify model weaknesses on particular subsets.

---

## 🚀 Installation & Setup

The project requires **Python 3.13+**. Environment and dependency management is handled via `pyproject.toml` (and `uv.lock` if using `uv`).

```bash
# Clone the repository
git clone <repository_url>
cd datathon2026

# Create a virtual environment and install the package
# (Using standard pip or uv)
pip install -e .

# Or if using uv:
uv sync
```

**Core Libraries:**
- `pandas` (>=3.0.2), `numpy`, `statsmodels`, `scikit-learn` (>=1.7.2)
- `lightgbm` (>=4.6.0), `xgboost` (>=3.2.0)
- `shap`, `matplotlib`, `seaborn`

---

## 📈 Usage (Workflow)

1. **Data Exploration:** Open `notebooks/eda.ipynb` to understand data distributions.
2. **Feature & Pipeline Testing:** Use `notebooks/base.ipynb` and `notebooks/time_feature.ipynb` to test the impact of features on the validation set (Train: 2013-2021, Valid: 2022).
3. **Running the Submission Pipeline:**
   - Open `notebooks/submission.ipynb`.
   - The pipeline will automatically merge features, initialize the `LightGBM` model, and train on the full historical dataset (2013 - 2022).
   - Generate predictions for the target horizon: `2023-01-01` to `2024-07-01`.
   - The final output is saved to the `submissions/miracle13.csv` directory.

---

## 📝 Optimization Notes (Next Steps)
- **Lag features:** Short-term lags like `lag_1` and `lag_7` perform excellently for COGS. However, mid-term lags like `lag_30` degrade the Revenue model due to added noise. Extreme caution is required when selecting lags.
- **`shock_period`:** This is a mandatory dummy variable that helps the model distinguish customer behavior before and after the severe drop in order value. Replacing the dummy with a continuous time distance (year distance) improves the validation score but significantly hurts performance on the test set.
