# 🏥 SoCal Patient Risk Stratification

A machine learning system that predicts 30-day hospital readmission risk for Southern California patients, calibrated to real California HCAI readmission rates. Refreshed quarterly with updated county-level data — no manual work after deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-automated-2088FF?logo=githubactions)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://gdiaz38-patient-risk-stratification.streamlit.app)**

---

## Overview

Hospital readmissions cost the US healthcare system over $26 billion annually. This project builds a patient risk stratification system for SoCal hospitals, combining real California HCAI county-level readmission rates with clinical and social determinants of health (SDOH) features to classify patients into four actionable risk tiers.

Key question it answers: *Which patients are most likely to be readmitted within 30 days — and what factors drive that risk?*

---

## Key Findings

- **Critical-tier patients** have an 8x higher readmission rate than Low-tier (36% vs 4%)
- **Insurance type** is the single strongest predictor — MediCare patients carry the highest risk
- **Los Angeles County** has the highest readmission rate among SoCal counties (15.6%)
- **Heart Failure** is the highest-risk diagnosis at 24% base readmission probability
- **SDOH score** ranks 4th in feature importance — social factors rival clinical severity

---

## Features

- **4-tier risk stratification** — Low, Medium, High, Critical based on percentile thresholds
- **County risk map** — SoCal scatter mapbox colored by readmission rate
- **Feature importance chart** — which clinical and SDOH factors drive predictions
- **Diagnosis heatmap** — risk score by age group and diagnosis combination
- **Insurance breakdown** — dual-axis risk score vs readmission rate by payer
- **Patient explorer** — searchable, filterable table with tier color coding
- **Quarterly auto-refresh** — GitHub Actions pulls latest HCAI rates and retrains

---

## Data Sources

| Source | Data | Update Frequency |
|---|---|---|
| [CA HCAI Open Data](https://data.chhs.ca.gov/) | County-level 30-day readmission rates (ICD-10) | Quarterly |
| Synthetic patient cohort | 8,000 patients calibrated to real HCAI rates | Regenerated quarterly |

Patient records are synthetic but statistically calibrated to real California readmission rates by county, age group, payer, and diagnosis.

---

## Project Structure

```
patient-risk-stratification/
├── .github/
│   └── workflows/
│       └── refresh.yml           # Quarterly cron — fetch rates, retrain, commit
├── app/
│   └── dashboard.py              # Streamlit dashboard
├── scripts/
│   ├── fetch_hcai.py             # CA HCAI open data API + local CSV fallback
│   ├── model.py                  # Patient generation + RF training + scoring
│   └── pipeline.py               # Orchestrates fetch → train → save
├── models/
│   └── risk_model.pkl            # Trained Random Forest + encoders
├── data/
│   ├── patients.csv              # 8,000 scored patients
│   ├── county_risk.csv           # County-level risk summary
│   ├── diagnosis_risk.csv        # Readmission rate by diagnosis
│   └── feature_importance.csv    # Model feature importances
└── requirements.txt
```

---

## How It Works

```
GitHub Actions (cron: 1st of Jan, Apr, Jul, Oct)
        ↓
fetch_hcai.py pulls latest CA readmission rates from CHHS API
Falls back to local ca_health.csv if API unavailable
        ↓
model.py generates 8,000 patients calibrated to updated rates
Applies SMOTE to fix class imbalance
Trains Random Forest (200 trees, balanced class weights)
Scores all patients → percentile-based risk tiers
        ↓
Saves patients.csv, county_risk.csv, diagnosis_risk.csv, risk_model.pkl
Commits → pushes to main
        ↓
Streamlit Community Cloud detects push → auto-redeploys
```

---

## ML Model

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 |
| Max depth | 8 |
| Class imbalance | SMOTE + balanced class weights |
| Risk tiers | Percentile-based (P25 / P50 / P75) |

**Features used:** age, length of stay, prior visits, distance to hospital, SDOH score, gender, county, insurance type, diagnosis code, age group

**Risk tier thresholds:**

| Tier | Risk Score | Readmission Rate |
|---|---|---|
| 🟢 Low | < P25 | ~4% |
| 🟡 Medium | P25–P50 | ~25% |
| 🟠 High | P50–P75 | ~37% |
| 🔴 Critical | > P75 | ~37% |

**Top risk factors by importance:**

| Factor | Importance |
|---|---|
| Insurance type | 22.4% |
| Age | 13.2% |
| Distance to hospital | 11.2% |
| SDOH score | 9.8% |
| Diagnosis code | 9.8% |

---

## Dashboard Sections

**KPI Row** — critical patient count, high-tier count, critical vs low readmission multiplier, highest-risk payer, highest-risk diagnosis

**Risk Tier Donut** — patient distribution across all 4 tiers

**Readmission by Tier** — bar chart showing actual readmission rates per tier

**County Risk Map** — SoCal scatter mapbox sized by patient count, colored by readmission rate

**Feature Importance** — horizontal bar chart of all 10 model features

**Diagnosis Risk** — readmission rate ranked by diagnosis

**Insurance Breakdown** — dual-axis avg risk score + readmission rate by payer type

**Age × Diagnosis Heatmap** — risk score grid across age groups and diagnoses

**Patient Explorer** — searchable by ID or county, filterable by tier, county, insurance, readmitted status

---

## Local Setup

### 1. Clone and create environment

```bash
git clone https://github.com/gdiaz38/patient-risk-stratification
cd patient-risk-stratification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run pipeline and launch dashboard

```bash
python3 scripts/pipeline.py
streamlit run app/dashboard.py
```

No API keys required.

---

## Deployment

### GitHub Actions (auto-refresh)

No secrets required. Workflow runs quarterly on the 1st of January, April, July, and October — pulls latest HCAI rates, retrains model, commits updated data.

### Streamlit Community Cloud

1. Connect repo at [share.streamlit.io](https://share.streamlit.io)
2. Main file: `app/dashboard.py`
3. Deploy — no secrets needed

---

## Tech Stack

`Python 3.11` · `Streamlit` · `Plotly` · `Scikit-learn` · `imbalanced-learn` · `Pandas` · `NumPy` · `GitHub Actions` · `CA HCAI Open Data`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT
