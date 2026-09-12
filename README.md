# 🚀 Quantitative AI & Machine Learning Portfolio

A B.Tech Mathematics & Computing portfolio featuring production-grade machine learning forecasting engines and advanced Generative AI orchestration pipelines.

---

## 📈 Project 1: Financial Time-Series Forecasting Engine

### 🔍 Objective
Built an end-to-end Machine Learning pipeline that extracts historical equity data, engineers technical mathematical indicators, and uses ensemble learning to forecast asset price closures.

### ⚙️ Technical Architecture & Workflow
- **Data Acquisition:** Automated ingestion of historical market data spanning 10 years via the Yahoo Finance API (`yfinance`).
- **Feature Engineering:** Developed custom statistical and quantitative signals using `Pandas` and `NumPy`, including:
  - 10-day Simple Moving Average (SMA)
  - 5-day Exponential Moving Average (EMA)
  - Daily Percentage Returns
  - Rolling Historical Volatility
- **Validation Strategy:** Implemented a strict **sequential chronological split** (80% Train / 20% Test) instead of a random train-test shuffle. This completely eliminates data leakage and preserves the temporal integrity required for time-series math.
- **Model Training:** Deployed a `Scikit-Learn` Random Forest Regressor optimized with 100 estimators.

### 📊 Performance Visualization
The model automatically outputs continuous predictions evaluated against baseline test trends. The saved evaluation visualization is archived locally as `prediction_chart.png`.
