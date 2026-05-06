import pandas as pd

df = pd.read_csv('ca_health.csv')

# Clean column names
df.columns = [c.strip() for c in df.columns]
print("Shape:", df.shape)
print("\nYears available:", sorted(df['Year'].unique()))
print("\nStrata types:", df['Strata'].unique())
print("\nSample counties:", df['County'].unique()[:15])
print("\nStrata Names:", df['Strata Name'].unique())

# Focus on ICD-10 data (2016+) which is more recent
df_icd10 = df[df['Total Admits (ICD-10)'].notna()].copy()
print(f"\nICD-10 rows: {len(df_icd10)}")

# Clean readmission rate
df_icd10['readmission_rate'] = df_icd10['30-day Readmission Rate (ICD-10)'].str.replace('%','').astype(float)

# Show readmission rates by strata
print("\nAvg Readmission Rate by Strata:")
print(df_icd10.groupby('Strata')['readmission_rate'].mean().round(2))

# Show SoCal counties
socal = ['Los Angeles','San Diego','Orange','Riverside','San Bernardino','Ventura']
print("\nSoCal County Readmission Rates:")
socal_df = df_icd10[df_icd10['County'].isin(socal)]
print(socal_df.groupby('County')['readmission_rate'].mean().round(2))
