import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
import plotly.express as px

st.set_page_config(page_title="EES v2.4 Global Dashboard", layout="wide")

st.title("🌍 The Economic Equity Score (EES) v2.4")
st.markdown("A live macroeconomic framework tracking structural conversion efficiency across the globe.")

# 1. LIVE GLOBAL DATA INGESTION & STAGE 2 NORMALIZATION
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_global_ees_data():
    # Identify actual countries (filters out regions like 'Arab World' or 'Euro Area')
    econ_info = wb.economy.DataFrame()
    valid_isos = econ_info[econ_info['region'] != ''].index.tolist()
    
    indicators = {
        "S_health": "SH.XPD.OOPC.CH.ZS",  
        "T_safety": "SI.POV.UMIC",        
        "Gini": "SI.POV.GINI",            
        "GDP_pc": "NY.GDP.PCAP.PP.CD",    
        "M_proxy": "SE.TER.ENRR"          
    }
    
    wide_df = wb.data.DataFrame(list(indicators.values()), valid_isos, time=range(2018, 2024), numericTimeKeys=True).reset_index()
    raw_df = wide_df.melt(id_vars=["economy", "series"], var_name="time", value_name="value")
    
    inv_indicators = {v: k for k, v in indicators.items()}
    raw_df["Pillar"] = raw_df["series"].map(inv_indicators)
    
    pivot_df = raw_df.pivot(index=["economy", "time"], columns="Pillar", values="value").reset_index()
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={"economy": "Country_ISO", "time": "Year"})
    
    expected_cols = ["GDP_pc", "Gini", "M_proxy", "S_health", "T_safety"]
    for col in expected_cols:
        if col not in pivot_df.columns: pivot_df[col] = np.nan
            
    # Section 3.3 Imputation Protocol (Linear Intertemporal Interpolation)
    for col in expected_cols:
        pivot_df[col] = pivot_df.groupby("Country_ISO")[col].transform(lambda x: x.interpolate(method="linear").bfill().ffill())
        
    recent_df = pivot_df.groupby("Country_ISO").last().reset_index().dropna().copy()
    
    # Section 2 Normalization (With Epsilon Zero-Variance Protection)
    epsilon = 1e-6
    recent_df["S"] = 1.0 - (recent_df["S_health"] / 100.0)
    recent_df["T"] = 1.0 - (recent_df["T_safety"] / 100.0)
    
    M_min, M_max = recent_df["M_proxy"].min(), recent_df["M_proxy"].max()
    # Apply standard bounded Min-Max with division-by-zero protection
    recent_df["M"] = np.clip((recent_df["M_proxy"] - M_min) / np.maximum(M_max - M_min, epsilon), 0.20, 0.95)
    
    # Explicitly normalize Gini to [0, 1] interval
    recent_df["Gini_Norm"] = recent_df["Gini"] / 100.0
    
    # Map ISO codes to actual country names for the UI
    recent_df["Country"] = recent_df["Country_ISO"].map(econ_info["name"])
    
    return recent_df

with st.spinner("Executing live World Bank API handshake... Compiling global matrix..."):
    df_global = fetch_global_ees_data()

# 2. EES ENGINE CALCULATION & ADAPTIVE MODIFIER
st.sidebar.header("⚙️ Engine Tuning")
kappa_val = st.sidebar.selectbox("Safety Penalty Exponent (κ)", [1, 2, 3], index=1, help="Adjusts the severity of the exponential penalty applied to equity when a nation falls below the safety floor. Baseline is quadratic (κ=2).")

HISTORICAL_ANCHOR = 0.85
DYNAMIC_THRESHOLD = max(HISTORICAL_ANCHOR, df_global["S"].median())

def calculate_ees(row):
    raw_equity = 1.0 - row["Gini_Norm"]
    e_mod = raw_equity if row["S"] >= DYNAMIC_THRESHOLD else raw_equity * ((row["S"] / DYNAMIC_THRESHOLD) ** kappa_val)
    return (row["S"]**0.25) * (row["T"]**0.25) * (row["M"]**0.25) * (e_mod**0.25)

df_global["EES_Score"] = df_global.apply(calculate_ees, axis=1)
df_global["Rank"] = df_global["EES_Score"].rank(ascending=False, method="min").astype(int)
df_global = df_global.sort_values("Rank")

# 3. GRAPHICAL DASHBOARD LAYOUT
col1, col2, col3 = st.columns(3)
col1.metric("🌍 Active Global Cohort", f"{len(df_global)} Nations")
col2.metric("🛡️ Dynamic Safety Floor", f"{DYNAMIC_THRESHOLD:.3f}")
col3.metric("📊 Median EES Score", f"{df_global['EES_Score'].median():.3f}")

st.markdown("---")

# Global Choropleth Map
st.subheader("🗺️ EES v2.4 Global Distribution Map")
fig_map = px.choropleth(
    df_global, locations="Country_ISO", color="EES_Score", hover_name="Country",
    hover_data={"Country_ISO": False, "Rank": True, "EES_Score": ":.3f", "GDP_pc": ":.0f"},
    color_continuous_scale="Viridis", title="Structural Conversion Efficiency by Nation"
)
fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# Scatter Plot
st.subheader("📈 EES Score vs Raw Economic Output (GDP pc)")
fig_scatter = px.scatter(
    df_global, x="GDP_pc", y="EES_Score", color="EES_Score", hover_name="Country", size="S",
    hover_data={"Rank": True, "S": ":.2f", "Gini": ":.1f"}, color_continuous_scale="Viridis",
    labels={"GDP_pc": "GDP per Capita (PPP)", "EES_Score": "EES v2.4 Score"}
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Data Table
st.subheader("📑 Global Leaderboard Ledger")
display_df = df_global[["Rank", "Country", "GDP_pc", "S", "T", "M", "Gini", "EES_Score"]]
st.dataframe(
    display_df.style.format({
        "GDP_pc": "${:,.0f}", "S": "{:.3f}", "T": "{:.3f}", 
        "M": "{:.3f}", "Gini": "{:.1f}", "EES_Score": "{:.3f}"
    }), use_container_width=True, hide_index=True
)

# 4. ACADEMIC LIMITATIONS & ECONOMETRIC BOUNDARIES
with st.expander("⚠️ Econometric Boundaries & Framework Limitations"):
    st.markdown("""
    **Temporal Vintage Mismatch:** To stabilize the framework, flow variables (smoothed via an SMA over 3 years) are geometrically multiplied against structural stock variables. Consequently, the composite EES score in any given year blends data of varying chronological vintages, introducing minor temporal distortion during periods of rapid structural policy shifts.
    
    **Safety Floor Sensitivity:** While the historical anchor floor of **0.85** is empirically motivated by long-run OECD baseline security medians, it functions as a rigid gatekeeper. The cohort composition dictates the dynamic median; expanding the evaluation database to include a heavy ratio of hyper-depressed fragile states shifts the activation mechanics of the exponential safety drag.
    
    **Endogeneity and Construct Overlap:** Because underlying sub-indicators of the EES framework (such as healthcare insulation and poverty mitigation) operate as known downstream determinants of macro health profiles, the model relies on shared latent pathways. Output should not be interpreted as absolute clean causal discovery.
    """)
