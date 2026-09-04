import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Configurare pagină
st.set_page_config(page_title="Soft Fotbal Pro", page_icon="⚽", layout="wide")

st.title("⚽ SOFT FOTBAL PRO - PREDICTOR XG & POISSON")
st.markdown("Aplicație web interactivă pentru analiza meciurilor de fotbal (Over/Under, GG & 1X2)")

# Sidebar - Introducere date meci
st.sidebar.header("⚙️ Configurare Meci")
liga = st.sidebar.selectbox("Selectează Liga", ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Altele"])
echipa_gazda = st.sidebar.text_input("Echipă Gazdă", "Arsenal")
echipa_oaspete = st.sidebar.text_input("Echipă Oaspete", "Chelsea")

col1, col2 = st.sidebar.columns(2)
with col1:
    xg_marcat_h = st.number_input("xG Marcate Gazdă", value=2.10, step=0.1)
    xg_primit_h = st.number_input("xG Primite Gazdă", value=0.85, step=0.1)
with col2:
    xg_marcat_a = st.number_input("xG Marcate Oaspete", value=1.45, step=0.1)
    xg_primit_a = st.number_input("xG Primite Oaspete", value=1.30, step=0.1)

cota_o25 = st.sidebar.number_input("Cotă Casa Over 2.5", value=1.95, step=0.05)
cota_gg = st.sidebar.number_input("Cotă Casa GG", value=1.85, step=0.05)

# Calcul Goluri Așteptate (Expected Goals)
exp_g_home = (xg_marcat_h + xg_primit_a) / 2
exp_g_away = (xg_marcat_a + xg_primit_h) / 2
total_exp_g = exp_g_home + exp_g_away

# Matrice Poisson (0-5 goluri)
max_goals = 6
poisson_matrix = np.zeros((max_goals, max_goals))
for i in range(max_goals):
    for j in range(max_goals):
        poisson_matrix[i, j] = poisson.pmf(i, exp_g_home) * poisson.pmf(j, exp_g_away)

# Calcul Probabilități
prob_home = float(np.sum(np.tril(poisson_matrix, -1)) * 100)
prob_draw = float(np.sum(np.diag(poisson_matrix)) * 100)
prob_away = float(np.sum(np.triu(poisson_matrix, 1)) * 100)

prob_o25 = float((1 - np.sum([poisson_matrix[i, j] for i in range(3) for j in range(3) if i + j < 3])) * 100)
prob_gg = float((1 - (np.sum(poisson_matrix[0, :]) + np.sum(poisson_matrix[:, 0]) - poisson_matrix[0,0])) * 100)

# Afișare rezultate principale
st.subheader(f"📊 Analiză: {echipa_gazda} vs {echipa_oaspete} ({liga})")

m1, m2, m3, m4 = st.columns(4)
m1.metric("xG Așteptat Gazdă", f"{exp_g_home:.2f}")
m2.metric("xG Așteptat Oaspete", f"{exp_g_away:.2f}")
m3.metric("Total xG Meci", f"{total_exp_g:.2f}")
m4.metric("Pronostic Principal", "OVER 2.5" if prob_o25 > 55 else "UNDER 2.5")

st.markdown("---")

# Recomandare Value Bet
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🎯 Probabilități Calculate")
    st.write(f"**Victorie {echipa_gazda} (1):** {prob_home:.1f}%")
    st.progress(min(int(prob_home), 100))
    st.write(f"**Egal (X):** {prob_draw:.1f}%")
    st.progress(min(int(prob_draw), 100))
    st.write(f"**Victorie {echipa_oaspete} (2):** {prob_away:.1f}%")
    st.progress(min(int(prob_away), 100))

with col_b:
    st.subheader("🔥 Indicator Value Bet")
    cota_reala_o25 = 100 / prob_o25 if prob_o25 > 0 else 0
    cota_reala_gg = 100 / prob_gg if prob_gg > 0 else 0
    
    val_o25 = cota_o25 * (prob_o25 / 100)
    val_gg = cota_gg * (prob_gg / 100)
    
    st.write(f"**Over 2.5:** Cotă Reală `{cota_reala_o25:.2f}` vs Cotă Casă `{cota_o25:.2f}`")
    if val_o25 > 1.05:
        st.success(f"🔥 VALUE BET DETECTAT Pe Over 2.5 (Margine: {((val_o25-1)*100):.1f}%)")
    else:
        st.info("Fără Valoare pe Over 2.5")
        
    st.write(f"**Ambele Marchează (GG):** Cotă Reală `{cota_reala_gg:.2f}` vs Cotă Casă `{cota_gg:.2f}`")
    if val_gg > 1.05:
        st.success(f"🔥 VALUE BET DETECTAT Pe GG (Margine: {((val_gg-1)*100):.1f}%)")
    else:
        st.info("Fără Valoare pe GG")
