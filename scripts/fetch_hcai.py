import requests, os, io
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# CA HCAI readmission data via CHHS open data portal
HCAI_URL = (
    "https://data.chhs.ca.gov/api/3/action/datastore_search"
    "?resource_id=9767cb68-8ea9-4f0b-8179-9431abc89f11&limit=5000"
)

SOCAL_COUNTIES = [
    "Los Angeles","San Diego","Orange",
    "Riverside","San Bernardino","Ventura"
]

def fetch_hcai() -> pd.DataFrame:
    """Pull latest CA hospital readmission rates from CHHS open data."""
    print("📥 Fetching HCAI readmission data...")
    try:
        r    = requests.get(HCAI_URL, timeout=30)
        data = r.json()["result"]["records"]
        df   = pd.DataFrame(data)
        print(f"   Fetched {len(df)} records from CHHS API")
        return df
    except Exception as e:
        print(f"⚠ HCAI API failed ({e}) — falling back to local ca_health.csv")
        return None

def build_county_rates(df_hcai=None) -> dict:
    """
    Build county → readmission rate mapping.
    Uses HCAI API if available, falls back to local CSV, then hardcoded defaults.
    """
    defaults = {
        "Los Angeles":    0.1558,
        "San Diego":      0.1419,
        "Orange":         0.1366,
        "Riverside":      0.1456,
        "San Bernardino": 0.1498,
        "Ventura":        0.1428,
    }

    # Try local CSV first (real HCAI data)
    local_path = os.path.join(os.path.dirname(__file__), "..", "ca_health.csv")
    alt_path   = os.path.join(os.path.dirname(__file__), "..", "ca_health1.csv")

    for path in [local_path, alt_path]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                df = df[df["30-day Readmission Rate (ICD-10)"].notna()].copy()
                df["rate"] = (df["30-day Readmission Rate (ICD-10)"]
                              .str.replace("%","", regex=False)
                              .pipe(pd.to_numeric, errors="coerce")
                              .div(100))
                county_rates = (df[df["County"].isin(SOCAL_COUNTIES)]
                                .groupby("County")["rate"]
                                .mean()
                                .to_dict())
                if county_rates:
                    print(f"   County rates from {os.path.basename(path)}:")
                    for c, r in county_rates.items():
                        print(f"     {c}: {r:.3f}")
                    return {**defaults, **county_rates}
            except Exception as e:
                print(f"⚠ Could not parse {path}: {e}")

    print("   Using hardcoded default rates")
    return defaults

if __name__ == "__main__":
    rates = build_county_rates()
    print(rates)
