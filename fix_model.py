import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load from PostgreSQL
engine = create_engine('postgresql://gabrieldiaz@localhost/patient_risk')
df = pd.read_sql('SELECT * FROM patients', engine)
print(f"Loaded {len(df)} patients")
print(f"Readmission rate: {df['readmitted_30d'].mean():.1%}")

# Feature engineering
le_dict = {}
df_model = df.copy()
for col in ['gender','county','insurance_type','diagnosis_code','age_group']:
    le = LabelEncoder()
    df_model[col+'_enc'] = le.fit_transform(df_model[col])
    le_dict[col] = le

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

# Fix class imbalance with SMOTE
print("\nApplying SMOTE to fix class imbalance...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Balanced training set: {y_train_bal.value_counts().to_dict()}")

# Retrain with balanced data + class weights
print("\nTraining improved Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)
preds = rf.predict(X_test)
proba = rf.predict_proba(X_test)[:,1]

print("\n📊 IMPROVED MODEL RESULTS")
print("="*40)
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, proba):.3f}")

# Fix risk tiers using percentiles instead of fixed thresholds
df['risk_score'] = rf.predict_proba(df_model[features])[:,1]
p25 = df['risk_score'].quantile(0.25)
p50 = df['risk_score'].quantile(0.50)
p75 = df['risk_score'].quantile(0.75)

df['risk_tier'] = pd.cut(
    df['risk_score'],
    bins=[0, p25, p50, p75, 1.0],
    labels=['Low','Medium','High','Critical'],
    include_lowest=True
)

risk_summary = df.groupby('risk_tier').agg(
    patient_count=('patient_id','count'),
    avg_risk_score=('risk_score','mean'),
    actual_readmission_rate=('readmitted_30d','mean')
).round(3)

print(f"\n🏥 IMPROVED RISK STRATIFICATION:")
print(risk_summary)

# Feature importance
importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 TOP RISK FACTORS:")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:25s} {bar} {row['importance']:.3f}")

# Export improved predictions for Tableau
export = df[['patient_id','age','age_group','gender','zip_code','county',
             'diagnosis_description','insurance_type','length_of_stay',
             'num_prior_visits','sdoh_score','readmitted_30d',
             'risk_score','risk_tier']].copy()
export['risk_score'] = export['risk_score'].round(4)
export.to_csv('tableau_patients.csv', index=False)

# County risk summary
county_summary = df.groupby('county').agg(
    avg_risk_score=('risk_score','mean'),
    readmission_rate=('readmitted_30d','mean'),
    patient_count=('patient_id','count'),
    high_risk_count=('risk_tier', lambda x: (x.isin(['High','Critical'])).sum())
).round(4).reset_index()
county_summary['high_risk_pct'] = (
    county_summary['high_risk_count'] / county_summary['patient_count']).round(3)
county_summary.to_csv('tableau_county_risk.csv', index=False)

# Diagnosis risk summary
diag_summary = df.groupby('diagnosis_description').agg(
    avg_risk_score=('risk_score','mean'),
    readmission_rate=('readmitted_30d','mean'),
    patient_count=('patient_id','count')
).round(4).reset_index().sort_values('avg_risk_score', ascending=False)
diag_summary.to_csv('tableau_diagnosis_risk.csv', index=False)

print("\n✅ Exported tableau_patients.csv")
print("✅ Exported tableau_county_risk.csv")
print("✅ Exported tableau_diagnosis_risk.csv")
print("\n🎯 Improved model complete!")
