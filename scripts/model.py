import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

from fetch_hcai import build_county_rates

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "risk_model.pkl")

np.random.seed(42)

DIAGNOSES = [
    ("I50", "Heart Failure",               0.24),
    ("J44", "COPD",                        0.19),
    ("N18", "Chronic Kidney Disease",      0.21),
    ("E11", "Type 2 Diabetes",             0.14),
    ("I21", "Acute Myocardial Infarction", 0.18),
    ("J18", "Pneumonia",                   0.16),
    ("I63", "Cerebral Infarction",         0.13),
    ("M54", "Back Pain",                   0.07),
    ("Z51", "Aftercare",                   0.09),
    ("S72", "Hip Fracture",                0.15),
]

AGE_RATES   = {"18-44": 0.08, "45-64": 0.138, "65+": 0.152}
PAYER_RATES = {
    "MediCare": 0.158, "Medi-Cal": 0.145,
    "Private":  0.098, "Self-Pay": 0.112, "Other": 0.121
}
COUNTY_ZIPS = {
    "Los Angeles":    ["90001","90011","90019","90026","90031"],
    "San Diego":      ["92101","92103","92108","92115","92120"],
    "Orange":         ["92602","92618","92651","92660","92705"],
    "Riverside":      ["92501","92503","92507","92530","92544"],
    "San Bernardino": ["92324","92335","92346","92354","92374"],
    "Ventura":        ["93001","93003","93010","93021","93030"],
}
SDOH_BASE = {
    "Los Angeles": 62, "San Diego": 48, "Orange": 38,
    "Riverside": 55, "San Bernardino": 60, "Ventura": 44
}

def generate_patients(county_rates: dict, n: int = 8000) -> pd.DataFrame:
    records = []
    for i in range(n):
        county    = np.random.choice(list(county_rates.keys()),
                                     p=[0.35,0.18,0.15,0.12,0.12,0.08])
        age_group = np.random.choice(["18-44","45-64","65+"], p=[0.20,0.38,0.42])
        age       = np.random.randint(*({"18-44":(18,45),"45-64":(45,65),"65+":(65,95)}[age_group]))
        gender    = np.random.choice(["Male","Female"], p=[0.48,0.52])
        insurance = np.random.choice(list(PAYER_RATES.keys()), p=[0.38,0.28,0.22,0.07,0.05])
        diag_code, diag_desc, diag_risk = DIAGNOSES[np.random.randint(0, len(DIAGNOSES))]

        los  = np.random.randint(1, 21)
        npv  = np.random.randint(0, 15)
        dist = round(np.random.exponential(8), 1)
        sdoh = round(np.clip(SDOH_BASE[county] + np.random.normal(0, 12), 0, 100), 1)

        prob = county_rates[county]
        prob *= PAYER_RATES[insurance] / 0.13
        prob *= AGE_RATES[age_group]   / 0.13
        prob *= (1 + diag_risk)
        prob *= (1 + 0.015 * npv)
        prob *= (1 + 0.008 * los)
        prob *= (1 + 0.003 * sdoh)
        prob  = float(np.clip(prob, 0.02, 0.95))

        records.append({
            "patient_id":            f"CA{i+10000:06d}",
            "age":                   age,
            "age_group":             age_group,
            "gender":                gender,
            "zip_code":              np.random.choice(COUNTY_ZIPS[county]),
            "county":                county,
            "diagnosis_code":        diag_code,
            "diagnosis_description": diag_desc,
            "insurance_type":        insurance,
            "length_of_stay":        los,
            "num_prior_visits":      npv,
            "distance_to_hospital":  dist,
            "sdoh_score":            sdoh,
            "readmitted_30d":        bool(np.random.binomial(1, prob)),
        })
    return pd.DataFrame(records)

def train(df: pd.DataFrame):
    le_dict  = {}
    df_model = df.copy()
    for col in ["gender","county","insurance_type","diagnosis_code","age_group"]:
        le = LabelEncoder()
        df_model[col+"_enc"] = le.fit_transform(df_model[col])
        le_dict[col] = le

    features = [
        "age","length_of_stay","num_prior_visits",
        "distance_to_hospital","sdoh_score",
        "gender_enc","county_enc","insurance_type_enc",
        "diagnosis_code_enc","age_group_enc"
    ]

    X = df_model[features]
    y = df_model["readmitted_30d"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_bal, y_bal)

    preds = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:,1]
    auc   = roc_auc_score(y_test, proba)
    print(f"\n📊 Model Results:")
    print(classification_report(y_test, preds))
    print(f"   ROC-AUC: {auc:.3f}")

    return rf, le_dict, features

def run():
    print("Loading county readmission rates...")
    county_rates = build_county_rates()

    print(f"\nGenerating 8,000 calibrated patients...")
    df = generate_patients(county_rates)
    print(f"  Readmission rate: {df['readmitted_30d'].mean():.1%}")

    print("\nTraining Random Forest with SMOTE...")
    rf, le_dict, features = train(df)

    # Score all patients
    df_model = df.copy()
    for col, le in le_dict.items():
        df_model[col+"_enc"] = le.transform(df_model[col])

    df["risk_score"] = rf.predict_proba(df_model[features])[:,1]
    p25, p50, p75    = df["risk_score"].quantile([0.25, 0.50, 0.75])
    df["risk_tier"]  = pd.cut(df["risk_score"],
                               bins=[0, p25, p50, p75, 1.0],
                               labels=["Low","Medium","High","Critical"],
                               include_lowest=True)

    risk_summary = df.groupby("risk_tier").agg(
        count=("patient_id","count"),
        avg_risk=("risk_score","mean"),
        readmission_rate=("readmitted_30d","mean")
    ).round(3)
    print(f"\n🏥 Risk Stratification:\n{risk_summary}")

    # Feature importance
    imp = pd.DataFrame({
        "feature":    features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n🔍 Top risk factors:")
    for _, row in imp.iterrows():
        print(f"  {row['feature']:25s} {row['importance']:.3f}")

    # Save outputs
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    export_cols = ["patient_id","age","age_group","gender","county",
                   "diagnosis_description","insurance_type","length_of_stay",
                   "num_prior_visits","sdoh_score","readmitted_30d",
                   "risk_score","risk_tier"]
    df[export_cols].to_csv(os.path.join(DATA_DIR, "patients.csv"), index=False)

    county_summary = df.groupby("county").agg(
        avg_risk=("risk_score","mean"),
        readmission_rate=("readmitted_30d","mean"),
        patient_count=("patient_id","count"),
        high_risk_pct=("risk_tier", lambda x: x.isin(["High","Critical"]).mean())
    ).round(4).reset_index()
    county_summary.to_csv(os.path.join(DATA_DIR, "county_risk.csv"), index=False)

    diag_summary = df.groupby("diagnosis_description").agg(
        avg_risk=("risk_score","mean"),
        readmission_rate=("readmitted_30d","mean"),
        count=("patient_id","count")
    ).round(4).reset_index().sort_values("avg_risk", ascending=False)
    diag_summary.to_csv(os.path.join(DATA_DIR, "diagnosis_risk.csv"), index=False)

    imp.to_csv(os.path.join(DATA_DIR, "feature_importance.csv"), index=False)

    pickle.dump({"model": rf, "encoders": le_dict, "features": features},
                open(MODEL_PATH, "wb"))

    print(f"\n✅ patients.csv        — {len(df)} rows")
    print(f"✅ county_risk.csv     — {len(county_summary)} counties")
    print(f"✅ diagnosis_risk.csv  — {len(diag_summary)} diagnoses")
    print(f"✅ risk_model.pkl      — saved")

if __name__ == "__main__":
    run()
