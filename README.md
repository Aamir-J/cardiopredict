# CardioPredict

**AI-Powered System for Early Heart Disease Risk Detection in Indian Patients**

An MBA capstone project at Dr. D. Y. Patil Vidyapeeth (DPU), facilitated through Qollabb / Stratix Project Advisory.

CardioPredict is a clinical decision-support prototype combining:
- A machine-learning risk model trained on real Indian-hospital data
- SHAP-based explainability for every prediction
- An interactive Streamlit dashboard
- HeartGuide — a conversational decision-support assistant powered by Claude

> **Important:** CardioPredict is a research prototype, not a diagnostic tool. It does not diagnose disease, prescribe treatment, or replace clinical judgement. All outputs are risk probability estimates intended to support, not replace, qualified clinical decision-making.

---

## Key results

- **Trained on 1,303 patients** — 1,000 from a Mendeley Indian multispecialty hospital dataset + 303 UCI Cleveland benchmark
- **XGBoost selected as winning model** — 96.5% recall, AUC-ROC 0.995, 5-fold CV recall 0.94 ± 0.02
- **Recall as primary metric** — medical false-negative cost framing
- **SHAP explainability** — feature contributions are clinically valid (ST slope, vessel involvement, BP)
- **Context-aware chatbot** — answers reference the specific patient's data, not generic advice

## Tech stack

| Layer | Tools |
|---|---|
| Data layer | Mendeley Indian-hospital CVD, UCI Cleveland |
| Storage | SQLite + SQLAlchemy |
| ML models | scikit-learn (LR, RF), XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Dashboard | Streamlit + Plotly |
| Chatbot | Anthropic Claude API |
| Deployment | Streamlit Community Cloud |

## Setup

```bash
# Clone
git clone https://github.com/<your-username>/cardiopredict.git
cd cardiopredict

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Download Mendeley dataset (CC BY 4.0)
# Visit https://data.mendeley.com/datasets/dzz48mvjht/1
# Save the CSV as data/raw/mendeley_cardiovascular.csv

# Run the pipeline
python -m src.ingest
python -m src.clean
python -m src.features
python -m src.train
python -m src.explain

# Launch the dashboard
python -m streamlit run app.py
```

## Repository layout
cardiopredict/
├── data/
│   └── raw/                  # Source CSVs (gitignored database files)
├── src/
│   ├── db.py                 # Database connection layer
│   ├── ingest.py             # Pulls raw datasets into SQLite
│   ├── clean.py              # Standardises and cleans
│   ├── features.py           # Feature engineering
│   ├── train.py              # Trains and compares models
│   └── explain.py            # SHAP analysis
├── models/                   # Trained model artefacts
├── dashboard/
│   ├── utils.py              # Model loading, prediction wrapper
│   ├── chatbot.py            # HeartGuide chatbot module
│   ├── page_risk_calculator.py
│   ├── page_analytics.py
│   ├── page_roi.py
│   └── page_metrics.py
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .env.example
└── README.md

## Dataset citations

**Mendeley:** Doppala, B.P. & Bhattacharyya, D. (2021). *Cardiovascular_Disease_Dataset*. Mendeley Data, V1. DOI: [10.17632/dzz48mvjht.1](https://doi.org/10.17632/dzz48mvjht.1)

**UCI:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X)

## Dataset audit note

A third dataset (Kaggle "Heart Attack Risk and Prediction Dataset in India", 10,000 records) was initially considered but **excluded** based on a synthetic-data audit:
- Zero missing values across all 260,000 cells (real medical data has natural missingness)
- Uniform distributions with means at exact range midpoints (signature of `numpy.random.uniform`)
- Biologically inconsistent lipid panel (LDL + HDL + Triglycerides/5 fails to approximate Total Cholesterol)
- No clinical source or institution named in the dataset description

See methodology chapter and `src/ingest.py` for the full audit rationale.

## License

This project is academic coursework. Code is released for review purposes; commercial reuse is not granted. Dataset licenses follow their original CC BY 4.0 terms.

## Disclaimer

This system is a research prototype. It is not approved by CDSCO, FDA, or any regulatory body as a medical device. It must not be used for clinical decision-making with real patients outside of supervised research contexts.