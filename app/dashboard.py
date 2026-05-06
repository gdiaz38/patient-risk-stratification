import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Patient Risk Stratification", page_icon="🏥", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

@st.cache_data(ttl=3600)
def load_data():
    patients  = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    county    = pd.read_csv(os.path.join(DATA_DIR, "county_risk.csv"))
    diagnosis = pd.read_csv(os.path.join(DATA_DIR, "diagnosis_risk.csv"))
    importance= pd.read_csv(os.path.join(DATA_DIR, "feature_importance.csv"))
    return patients, county, diagnosis, importance

patients, county, diagnosis, importance = load_data()

TIER_COLORS = {
    "Low":      "#2DC653",
    "Medium":   "#F4D03F",
    "High":     "#F77F00",
    "Critical": "#E63946",
}
TIER_ORDER = ["Low", "Medium", "High", "Critical"]

# ── Header ────────────────────────────────────────────────────────────────
st.title("🏥 SoCal Patient Risk Stratification")
st.caption(
    f"Random Forest classifier • {len(patients):,} patients • "
    f"Calibrated to real CA HCAI readmission rates • "
    f"Readmission rate: {patients['readmitted_30d'].mean():.1%}"
)

# ── KPIs ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

critical = patients[patients["risk_tier"] == "Critical"]
high     = patients[patients["risk_tier"] == "High"]
top_ins  = patients.groupby("insurance_type")["risk_score"].mean().idxmax()
top_diag = diagnosis.iloc[0]["diagnosis_description"]
critical_readmit = critical["readmitted_30d"].mean()
low_readmit      = patients[patients["risk_tier"]=="Low"]["readmitted_30d"].mean()

k1.metric("🔴 Critical Tier",    f"{len(critical):,} patients",
          f"{len(critical)/len(patients):.0%} of total")
k2.metric("🟠 High Tier",        f"{len(high):,} patients",
          f"{len(high)/len(patients):.0%} of total")
k3.metric("⚡ Critical vs Low",  f"{critical_readmit/low_readmit:.1f}x higher",
          "readmission rate")
k4.metric("💳 Highest Risk Payer", top_ins)
k5.metric("🏷 Highest Risk Dx",   top_diag.split()[0] + "...")

st.divider()

# ── Row 1: Risk tier donut + county map ───────────────────────────────────
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Risk Tier Distribution")
    tier_counts = (patients["risk_tier"]
                   .value_counts()
                   .reindex(TIER_ORDER)
                   .reset_index())
    tier_counts.columns = ["tier", "count"]

    fig1 = px.pie(
        tier_counts, names="tier", values="count",
        color="tier", color_discrete_map=TIER_COLORS,
        hole=0.55,
        category_orders={"tier": TIER_ORDER}
    )
    fig1.update_traces(textposition="outside", textinfo="percent+label")
    fig1.update_layout(height=320, showlegend=False,
                       margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig1, use_container_width=True)

    # Readmission rate by tier
    tier_readmit = (patients.groupby("risk_tier")["readmitted_30d"]
                    .mean().reindex(TIER_ORDER).reset_index())
    tier_readmit.columns = ["tier","rate"]
    fig1b = px.bar(
        tier_readmit, x="tier", y="rate",
        color="tier", color_discrete_map=TIER_COLORS,
        text=tier_readmit["rate"].apply(lambda v: f"{v:.0%}"),
        labels={"rate":"Readmission Rate","tier":""},
        category_orders={"tier": TIER_ORDER}
    )
    fig1b.update_traces(textposition="outside")
    fig1b.update_layout(height=260, showlegend=False,
                        yaxis_tickformat=".0%", yaxis_title="30-day Readmission Rate")
    st.plotly_chart(fig1b, use_container_width=True)

with col2:
    st.subheader("County Risk Map — SoCal")
    county_coords = {
        "Los Angeles":    (34.05, -118.24),
        "San Diego":      (32.72, -117.15),
        "Orange":         (33.70, -117.83),
        "Riverside":      (33.98, -117.37),
        "San Bernardino": (34.10, -117.29),
        "Ventura":        (34.27, -119.22),
    }
    county["lat"] = county["county"].map(lambda c: county_coords[c][0])
    county["lon"] = county["county"].map(lambda c: county_coords[c][1])
    county["readmission_pct"] = (county["readmission_rate"] * 100).round(1)

    fig2 = px.scatter_mapbox(
        county,
        lat="lat", lon="lon",
        color="readmission_pct",
        size="patient_count",
        hover_name="county",
        hover_data={"readmission_pct": True, "avg_risk": True,
                    "high_risk_pct": True, "lat": False, "lon": False},
        color_continuous_scale="RdYlGn_r",
        size_max=45, zoom=7,
        center={"lat": 33.8, "lon": -117.8},
        mapbox_style="carto-positron",
        labels={"readmission_pct": "Readmission %",
                "avg_risk": "Avg Risk Score",
                "high_risk_pct": "High/Critical %"}
    )
    fig2.update_layout(height=600, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Feature importance + diagnosis risk ────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Top Risk Factors")
    imp_sorted = importance.sort_values("importance")
    fig3 = px.bar(
        imp_sorted, x="importance", y="feature",
        orientation="h",
        color="importance", color_continuous_scale="Reds",
        text=imp_sorted["importance"].apply(lambda v: f"{v:.3f}"),
        labels={"importance": "Feature Importance", "feature": ""}
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(height=380, coloraxis_showscale=False,
                       yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Readmission Rate by Diagnosis")
    diag_plot = diagnosis.sort_values("readmission_rate", ascending=True)
    fig4 = px.bar(
        diag_plot, x="readmission_rate", y="diagnosis_description",
        orientation="h",
        color="readmission_rate", color_continuous_scale="RdYlGn_r",
        text=diag_plot["readmission_rate"].apply(lambda v: f"{v:.0%}"),
        labels={"readmission_rate": "Readmission Rate", "diagnosis_description": ""}
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(height=380, coloraxis_showscale=False,
                       xaxis_tickformat=".0%")
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Insurance breakdown + age/risk heatmap ────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Risk Score by Insurance Type")
    ins_df = (patients.groupby("insurance_type")
              .agg(avg_risk=("risk_score","mean"),
                   readmission_rate=("readmitted_30d","mean"),
                   count=("patient_id","count"))
              .reset_index()
              .sort_values("avg_risk", ascending=False))

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=ins_df["insurance_type"], y=ins_df["avg_risk"],
        name="Avg Risk Score", marker_color="#E63946", yaxis="y"
    ))
    fig5.add_trace(go.Scatter(
        x=ins_df["insurance_type"],
        y=ins_df["readmission_rate"],
        name="Readmission Rate", mode="lines+markers",
        marker=dict(size=10, color="#00B4D8"),
        line=dict(color="#00B4D8", width=2), yaxis="y2"
    ))
    fig5.update_layout(
        height=320,
        yaxis=dict(title="Avg Risk Score"),
        yaxis2=dict(title="Readmission Rate", overlaying="y",
                    side="right", tickformat=".0%"),
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified"
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("Risk Score by Age Group & Diagnosis")
    heat_df = (patients.groupby(["age_group","diagnosis_description"])
               ["risk_score"].mean().reset_index())
    heat_pivot = heat_df.pivot(index="diagnosis_description",
                                columns="age_group",
                                values="risk_score").round(3)
    fig6 = px.imshow(
        heat_pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels={"color":"Avg Risk Score"},
        text_auto=".2f"
    )
    fig6.update_layout(height=320,
                       xaxis_title="Age Group",
                       yaxis_title="")
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── Patient table ─────────────────────────────────────────────────────────
st.subheader("🔍 Patient Risk Explorer")

sc, f1, f2, f3, f4 = st.columns([2, 1, 1, 1, 1])
with sc:
    search = st.text_input("Search patient ID or county",
                           placeholder="e.g. CA010045, Los Angeles",
                           label_visibility="collapsed")
with f1:
    tier_filter = st.selectbox("Risk tier", ["All"] + TIER_ORDER,
                               label_visibility="collapsed")
with f2:
    county_filter = st.selectbox("County", ["All"] + sorted(patients["county"].unique()),
                                 label_visibility="collapsed")
with f3:
    ins_filter = st.selectbox("Insurance", ["All"] + sorted(patients["insurance_type"].unique()),
                               label_visibility="collapsed")
with f4:
    readmit_only = st.checkbox("Readmitted only", value=False)

filtered = patients.copy()
if search:
    filtered = filtered[
        filtered["patient_id"].str.contains(search, case=False) |
        filtered["county"].str.contains(search, case=False)
    ]
if tier_filter   != "All": filtered = filtered[filtered["risk_tier"] == tier_filter]
if county_filter != "All": filtered = filtered[filtered["county"] == county_filter]
if ins_filter    != "All": filtered = filtered[filtered["insurance_type"] == ins_filter]
if readmit_only:           filtered = filtered[filtered["readmitted_30d"] == True]

st.caption(f"Showing {len(filtered):,} of {len(patients):,} patients")

display_cols = {
    "patient_id":            "Patient ID",
    "age":                   "Age",
    "age_group":             "Age Group",
    "gender":                "Gender",
    "county":                "County",
    "diagnosis_description": "Diagnosis",
    "insurance_type":        "Insurance",
    "length_of_stay":        "LOS (days)",
    "num_prior_visits":      "Prior Visits",
    "sdoh_score":            "SDOH Score",
    "risk_score":            "Risk Score",
    "risk_tier":             "Risk Tier",
    "readmitted_30d":        "Readmitted",
}

def color_tier(val):
    colors = {"Low":"#1a472a","Medium":"#856404","High":"#7c3800","Critical":"#6b1a1a"}
    return f"background-color: {colors.get(val,'')}; color: white" if val in colors else ""

show = (filtered[list(display_cols.keys())]
        .rename(columns=display_cols)
        .reset_index(drop=True))

st.dataframe(
    show.style.applymap(color_tier, subset=["Risk Tier"]),
    use_container_width=True, height=450
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown(f"**Total patients:** {len(patients):,}")
    st.markdown(f"**Readmission rate:** {patients['readmitted_30d'].mean():.1%}")
    st.markdown(f"**Counties:** {patients['county'].nunique()}")
    st.markdown(f"**Diagnoses:** {patients['diagnosis_description'].nunique()}")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("- 🌲 Random Forest (200 trees)")
    st.markdown("- ⚖️ SMOTE + balanced weights")
    st.markdown("- 📊 Percentile-based tiers")
    st.markdown("---")
    st.markdown("**Data Source**")
    st.markdown("- CA HCAI readmission rates")
    st.markdown("- 8,000 calibrated patients")
    st.markdown("---")
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()
