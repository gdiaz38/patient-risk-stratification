import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Real CA readmission rates by county ─────────────────────────────────
county_rates = {
    'Los Angeles':    0.1558,
    'San Diego':      0.1419,
    'Orange':         0.1366,
    'Riverside':      0.1456,
    'San Bernardino': 0.1498,
    'Ventura':        0.1428,
}

# ── Real CA readmission rates by age group ───────────────────────────────
age_rates = {
    '18-44':  0.08,
    '45-64':  0.138,
    '65+':    0.152,
}

# ── Real CA readmission rates by payer ───────────────────────────────────
payer_rates = {
    'MediCare':  0.158,
    'Medi-Cal':  0.145,
    'Private':   0.098,
    'Self-Pay':  0.112,
    'Other':     0.121,
}

# ── SoCal zip codes by county ────────────────────────────────────────────
county_zips = {
    'Los Angeles':    ['90001','90011','90019','90026','90031','90044','90057','90063'],
    'San Diego':      ['92101','92103','92108','92115','92120','92126','92131','92139'],
    'Orange':         ['92602','92618','92651','92660','92705','92780','92801','92868'],
    'Riverside':      ['92501','92503','92507','92530','92544','92553','92571','92584'],
    'San Bernardino': ['92324','92335','92346','92354','92374','92376','92401','92408'],
    'Ventura':        ['93001','93003','93010','93021','93030','93033','93060','93063'],
}

# ── Diagnosis codes (common readmission conditions) ──────────────────────
diagnoses = [
    ('I50',  'Heart Failure',              0.24),
    ('J44',  'COPD',                       0.19),
    ('N18',  'Chronic Kidney Disease',     0.21),
    ('E11',  'Type 2 Diabetes',            0.14),
    ('I21',  'Acute Myocardial Infarction',0.18),
    ('J18',  'Pneumonia',                  0.16),
    ('I63',  'Cerebral Infarction',        0.13),
    ('M54',  'Back Pain',                  0.07),
    ('Z51',  'Aftercare',                  0.09),
    ('S72',  'Hip Fracture',               0.15),
]

# ── Generate 8,000 synthetic patients ────────────────────────────────────
n = 8000
records = []

for _ in range(n):
    # Demographics
    county = np.random.choice(list(county_rates.keys()),
                              p=[0.35,0.18,0.15,0.12,0.12,0.08])
    age_group = np.random.choice(['18-44','45-64','65+'], p=[0.20,0.38,0.42])
    if age_group == '18-44':
        age = np.random.randint(18, 45)
    elif age_group == '45-64':
        age = np.random.randint(45, 65)
    else:
        age = np.random.randint(65, 95)

    gender = np.random.choice(['Male','Female'], p=[0.48,0.52])
    zip_code = np.random.choice(county_zips[county])
    insurance = np.random.choice(list(payer_rates.keys()),
                                 p=[0.38,0.28,0.22,0.07,0.05])

    # Clinical features
    diag = diagnoses[np.random.randint(0, len(diagnoses))]
    diag_code, diag_desc, diag_risk = diag
    length_of_stay   = np.random.randint(1, 21)
    num_prior_visits = np.random.randint(0, 15)
    distance_to_hosp = round(np.random.exponential(8), 1)

    # SDOH score (0-100, higher = more vulnerable)
    sdoh_base = {'Los Angeles':62,'San Diego':48,'Orange':38,
                 'Riverside':55,'San Bernardino':60,'Ventura':44}
    sdoh_score = round(np.clip(
        sdoh_base[county] + np.random.normal(0, 12), 0, 100), 1)

    # Readmission probability (combining all real risk factors)
    base_prob = county_rates[county]
    base_prob *= (payer_rates[insurance] / 0.13)
    base_prob *= (age_rates[age_group]   / 0.13)
    base_prob *= (1 + diag_risk)
    base_prob *= (1 + 0.015 * num_prior_visits)
    base_prob *= (1 + 0.008 * length_of_stay)
    base_prob *= (1 + 0.003 * sdoh_score)
    base_prob  = np.clip(base_prob, 0.02, 0.95)
    readmitted = np.random.binomial(1, base_prob)

    records.append({
        'patient_id':           f'CA{_+10000:06d}',
        'age':                  age,
        'age_group':            age_group,
        'gender':               gender,
        'zip_code':             zip_code,
        'county':               county,
        'diagnosis_code':       diag_code,
        'diagnosis_description':diag_desc,
        'insurance_type':       insurance,
        'length_of_stay':       length_of_stay,
        'num_prior_visits':     num_prior_visits,
        'distance_to_hospital': distance_to_hosp,
        'sdoh_score':           sdoh_score,
        'readmitted_30d':       bool(readmitted),
    })

df = pd.DataFrame(records)
print(f"Dataset: {len(df)} patients")
print(f"Readmission rate: {df['readmitted_30d'].mean():.1%}")
print(f"\nReadmission by county:")
print(df.groupby('county')['readmitted_30d'].mean().sort_values(ascending=False).round(3))

# ── Load into PostgreSQL ─────────────────────────────────────────────────
engine = create_engine('postgresql://gabrieldiaz@localhost/patient_risk')
df.to_sql('patients', engine, if_exists='replace', index=False)
print("\n✅ Patient data loaded into PostgreSQL!")

# ── Feature engineering ──────────────────────────────────────────────────
le = LabelEncoder()
df_model = df.copy()
for col in ['gender','county','insurance_type','diagnosis_code','age_group']:
    df_model[col+'_enc'] = le.fit_transform(df_model[col])

features = [
    'age','length_of_stay','num_prior_visits',
    'distance_to_hospital','sdoh_score',
    'gender_enc','county_enc','insurance_type_enc',
    'diagnosis_code_enc','age_group_enc'
]
X = df_model[features]
y = df_model['readmitted_30d'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train Random Forest ───────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]

print("\n📊 RANDOM FOREST RESULTS")
print("="*40)
print(classification_report(y_test, rf_preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_proba):.3f}")

# ── Train XGBoost ─────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:,1]

print("\n📊 XGBOOST RESULTS")
print("="*40)
print(classification_report(y_test, xgb_preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_proba):.3f}")

# ── Feature importance ────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 TOP RISK FACTORS:")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:25s} {bar} {row['importance']:.3f}")

# ── Risk stratification ───────────────────────────────────────────────────
df['risk_score']  = rf.predict_proba(df_model[features])[:,1]
df['risk_tier']   = pd.cut(df['risk_score'],
                            bins=[0, 0.25, 0.50, 0.75, 1.0],
                            labels=['Low','Medium','High','Critical'])

risk_summary = df.groupby('risk_tier').agg(
    patient_count=('patient_id','count'),
    avg_risk_score=('risk_score','mean'),
    actual_readmission_rate=('readmitted_30d','mean')
).round(3)
print(f"\n🏥 RISK STRATIFICATION SUMMARY:")
print(risk_summary)

# ── Save predictions for Tableau ─────────────────────────────────────────
export = df[['patient_id','age','age_group','gender','zip_code','county',
             'diagnosis_description','insurance_type','length_of_stay',
             'num_prior_visits','sdoh_score','readmitted_30d',
             'risk_score','risk_tier']].copy()
export['risk_score'] = export['risk_score'].round(4)
export.to_csv('tableau_patients.csv', index=False)

# County risk summary for map
county_summary = df.groupby('county').agg(
    avg_risk_score=('risk_score','mean'),
    readmission_rate=('readmitted_30d','mean'),
    patient_count=('patient_id','count'),
    high_risk_count=('risk_tier', lambda x: (x.isin(['High','Critical'])).sum())
).round(4).reset_index()
county_summary['high_risk_pct'] = (
    county_summary['high_risk_count'] / county_summary['patient_count']).round(3)
county_summary.to_csv('tableau_county_risk.csv', index=False)

print("\n✅ Exported tableau_patients.csv")
print("✅ Exported tableau_county_risk.csv")
print("\n🎯 Model training complete!")

