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
col1, col2, col
