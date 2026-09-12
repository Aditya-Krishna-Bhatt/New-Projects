# 🚀 Advanced Quantitative AI & Machine Learning Portfolio

A specialized asset portfolio leveraging a strong foundational background in **B.Tech Mathematics & Computing** to engineer production-grade machine learning pipelines and token-optimized Generative AI orchestration architectures.

---

## 📈 Project 1: Financial Time-Series Forecasting Engine
**Tech Stack:** `Python`, `Scikit-Learn`, `Pandas`, `NumPy`, `Matplotlib`, `yFinance`

### 🔍 Core Problem & Logic
Built an end-to-end Machine Learning prediction framework designed to ingest financial equities, engineer technical indicators, and run ensemble models to predict asset price movements.

### ⚙️ Engineering Workflow
- **Data Ingestion Pipeline:** Automated historical equity data mining via the Yahoo Finance framework.
- **Mathematical Feature Engineering:** Handled dense numerical arrays to build quantitative features, including 10-day Simple Moving Average (SMA), 5-day Exponential Moving Average (EMA), daily fractional returns, and a rolling standard deviation volatility index.
- **Data Leakage Mitigation:** Implemented a strict **chronological sequential split** (80% Train / 20% Test) rather than a random shuffle. This critical design preserves the temporal mathematical integrity required for valid time-series validation.
- **Predictive Modelling:** Deployed a Scikit-Learn Random Forest Regressor optimized with 100 estimators.

---

## 🤖 Project 2: Free Intelligent Document Assistant (RAG Pipeline)
**Tech Stack:** `Python`, `Meta Llama 3.3`, `Groq Cloud API`, `Streamlit`, `PyPDF`

### 🔍 Core Problem & Logic
Architected a highly scalable, hardware-agnostic Retrieval-Augmented Generation (RAG) system capable of parsing unstructured document matrices (such as academic volumes) and delivering factually grounded conversational insights.

### ⚙️ Engineering Workflow
- **Lean Extraction Architecture:** Re-engineered the text loader pipeline using `PyPDF` directly, avoiding heavy legacy AI framework dependencies to remain stable under advanced execution runtimes.
- **Token Optimization & Budget Guardrails:** Implemented a lightweight native keyword-matching extraction index. By isolating only the top relevant text segments based on user query tokens, the input payload was dropped to roughly 1,500 tokens, successfully bypassing cloud rate limit rejections (Error 413) on free service tiers.
- **Orchestration Layer:** Structured deterministic prompt engineering paradigms to isolate context boundaries and prevent LLM hallucinations. Integrated Meta's Llama 3.3 model via Groq's high-performance cloud clusters.
- **User Interface Deployment:** Wrapped the backend processing inside an interactive web browser dashboard utilizing **Streamlit**.

