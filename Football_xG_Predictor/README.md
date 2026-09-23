# ⚽ AI-Driven Expected Goals (xG) Shot Predictor

An end-to-end Machine Learning web application that calculates the statistical probability of a football shot resulting in a goal based on spatial physics, coordinate geometry, and contextual match dynamics(circumstances of an attempt).

## 📊 Live Application
👉 [Click Here to Interact with the Live Dashboard](INSERT_YOUR_STREAMLIT_COMMUNITY_CLOUD_LINK_HERE)

## 📌 Project Overview
Traditional football metrics fail to evaluate the true quality of goal-scoring opportunities. This project frames shot evaluation as a **supervised binary classification task**. Utilizing a dataset of **21,153 historical shots** harvested from StatsBomb's open repository, an extreme gradient-boosting framework was engineered to map field geometry directly to historical conversion rates. Note: Only LaLiga stats were used to train the model.

### Key Metrics Engineered:
- **Euclidean Shot Distance:** Precise diagonal distance calculated from player coordinates \((x, y)\) to the center of the goals \((120, 40)\).
- **Subtended Shot Angle:** Vector tracking using dot products to isolate the open visual shooting angle to both defensive goalposts.

## ⚙️ Model Performance & Diagnostics
The underlying predictive engine was built using **XGBoost** and validated using robust probabilistic metrics suited for highly imbalanced tracking datasets:

- **ROC-AUC Score:** `0.8011` (Reflects exceptional discriminatory capacity between goals and misses).
- **Brier Score Loss:** `0.0889` (Proves precise calibration of output percentage assignments).

### Feature Dominance Hierarchy:
1. Contextual Play Pattern (`play_pattern_Other` - Free-kicks/Rebounds): **29.53%**
2. Visual Goal Angle Open to Shooter (`angle`): **12.38%**
3. Defensive Pressure Proximity (`under_pressure`): **10.75%**

## 🛠️ Tech Stack & Architecture
- **Language:** Python 3.11+
- **ML Framework:** XGBoost, Scikit-Learn
- **Data Engineering:** Pandas, NumPy
- **Interface & Tooling:** Streamlit, JSON Serialization

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install the necessary virtual environment dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app_xG_predictor.py
   ```
