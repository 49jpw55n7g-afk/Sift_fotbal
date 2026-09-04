import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

st.set_page_config(page_title="Predictor Complete Superbet", page_icon="⚽", layout="wide")

st.title("⚽ PREDICTOR COMPLET - TOATE PIEȚELE SUPERBET")

# Sidebar - Configurare & Conectare
st.sidebar.header("⚙️ Setări Meci")
api_key = st.sidebar.text_input("Cheie API Football-Data.org", value="20505c2f8aaa48e58a6c4764d0664e7f", type="password")

@st.cache_data(ttl=3600)
def fetch_matches(api_key):
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": api_key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("matches", [])
        return []
    except:
        return []

matches = fetch_matches(api_key)

if matches:
    match_options = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['competition']['name']})": m for m in matches}
    
    # Salvarea opțiunii selectate în session_state pentru a preveni săritul la primul meci
    if "selected_match_key" not in st.session_state:
        st.session_state.selected_match_key = list(match_options.keys())[0]

    selected_match_name = st.selectbox(
        "Alege Meciul Zilei", 
        list(match_options.keys()), 
        key="selected_match_key"
    )
    
    selected_match = match_options[selected_match_name]
    echipa_gazda = selected_match['homeTeam']['name']
    echipa_oaspete = selected_match['awayTeam']['name']
else:
    st.info("Folosiți introducerea manuală dacă meciurile nu se încarcă automat.")
    echipa_gazda = st.sidebar.text_input("Echipă Gazdă", "Arsenal")
    echipa_oaspete = st.sidebar.text_input("Echipă Oaspete", "Chelsea")

# Parametri Introduși (Modificarea lor nu va mai reseta meciul)
st.sidebar.subheader("📊 Media de Goluri / xG")
exp_g_home = st.sidebar.number_input("xG Gazdă", value=1.70, step=0.1, key="xg_h")
exp_g_away = st.sidebar.number_input("xG Oaspete", value=1.20, step=0.1, key="xg_a")

st.sidebar.subheader("🟨 Cartonașe & 🚩 Cornere")
medie_cartonase = st.sidebar.number_input("Medie Cartonașe / Meci", value=4.5, step=0.5, key="cart")
medie_cornere = st.sidebar.number_input("Medie Cornere / Meci", value=9.5, step=0.5, key="corn")

# ---------------------------------------------------------
# CALCUL MATEMATIC COMPLET
# ---------------------------------------------------------
max_goals = 6

exp_r1_h, exp_r1_a = exp_g_home * 0.43, exp_g_away * 0.43

def build_poisson_matrix(exp_h, exp_a):
    mat = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            mat[i, j] = poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a)
    return mat

mat_full = build_poisson_matrix(exp_g_home, exp_g_away)
mat_r1 = build_poisson_matrix(exp_r1_h, exp_r1_a)

# Probabilități Final 1X2
p1 = float(np.sum(np.tril(mat_full, -1)) * 100)
px = float(np.sum(np.diag(mat_full)) * 100)
p2 = float(np.sum(np.triu(mat_full, 1)) * 100)

# Probabilități Repriza 1
p1_r1 = float(np.sum(np.tril(mat_r1, -1)) * 100)
px_r1 = float(np.sum(np.diag(mat_r1)) * 100)
p2_r1 = float(np.sum(np.triu(mat_r1, 1)) * 100)

# Pauză sau Final (PsF)
psf1 = min(100.0, p1 + p1_r1 - (p1 * p1_r1 / 100))
psfx = min(100.0, px + px_r1 - (px * px_r1 / 100))
psf2 = min(100.0, p2 + p2_r1 - (p2 * p2_r1 / 100))

# Goluri
p_under15 = float(np.sum([mat_full[i, j] for i in range(2) for j in range(2) if i + j < 2]) * 100)
p_under25 = float(np.sum([mat_full[i, j] for i in range(3) for j in range(3) if i + j < 3]) * 100)
p_under35 = float(np.sum([mat_full[i, j] for i in range(4) for j in range(4) if i + j < 4]) * 100)

p_over15 = 100 - p_under15
p_over25 = 100 - p_under25
p_over35 = 100 - p_under35

p_gg = float((1 - (np.sum(mat_full[0, :]) + np.sum(mat_full[:, 0]) - mat_full[0,0])) * 100)
p_gg_r1 = float((1 - (np.sum(mat_r1[0, :]) + np.sum(mat_r1[:, 0]) - mat_r1[0,0])) * 100)

p_g13 = float(np.sum([mat_full[i, j] for i in range(4) for j in range(4) if 1 <= i + j <= 3]) * 100)
p_g24 = float(np.sum([mat_full[i, j] for i in range(5) for j in range(5) if 2 <= i + j <= 4]) * 100)

# Cartonașe & Cornere
p_cart_over35 = (1 - poisson.cdf(3, medie_cartonase)) * 100
p_cart_over45 = (1 - poisson.cdf(4, medie_cartonase)) * 100
p_corn_over85 = (1 - poisson.cdf(8, medie_cornere)) * 100
p_corn_over95 = (1 - poisson.cdf(9, medie_cornere)) * 100

# Centralizare pariuri
toate_pariurile = {
    f"Șansă Dublă: 1X ({echipa_gazda} sau Egal)": p1 + px,
    f"Șansă Dublă: X2 (Egal sau {echipa_oaspete})": px + p2,
    f"Șansă Dublă: 12 (Fără Egal)": p1 + p2,
    f"Pauză sau Final: PsF 1": psf1,
    f"Pauză sau Final: PsF X": psfx,
    f"Pauză sau Final: PsF 2": psf2,
    "Goluri: Peste 1.5 Goluri": p_over15,
    "Goluri: Sub 3.5 Goluri": p_under35,
    "Goluri: Sub 2.5 Goluri": p_under25,
    "Goluri: Peste 2.5 Goluri": p_over25,
    "Goluri: Ambele Marchează (GG)": p_gg,
    "Interval Goluri: 1-3 Goluri": p_g13,
    "Interval Goluri: 2-4 Goluri": p_g24,
    "Cornere: Peste 8.5 Cornere": p_corn_over85,
    "Cartonașe: Peste 3.5 Cartonașe": p_cart_over35,
    f"Solist: Victorie {echipa_gazda} (1)": p1,
    f"Solist: Victorie {echipa_oaspete} (2)": p2
}

pariuri_sortate = sorted(toate_pariurile.items(), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# INTERFAȚĂ PE TAB-URI
# ---------------------------------------------------------
st.subheader(f"🏟️ {echipa_gazda} vs {echipa_oaspete}")

tab_top, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Top Șanse (Pariul Zilei)",
    "🎯 Principale & PsF", 
    "⚽ Total Goluri & Intervale", 
    "⏱️ Repriza 1", 
    "🔥 Ambele Marchează (GG)", 
    "🟨 Cartonașe", 
    "🚩 Cornere"
])

with tab_top:
    st.markdown("### 🔥 Cele mai sigure opțiuni (Șanse de reușită calculate)")
    st.write("Iată pariurile cu cea mai mare probabilitate matematică pentru acest meci:")
    
    col_a, col_b, col_c = st.columns(3)
    
    pariul_1, prob_1 = pariuri_sortate[0]
    pariul_2, prob_2 = pariuri_sortate[1]
    pariul_3, prob_3 = pariuri_sortate[2]
    
    with col_a:
        st.success(f"🥇 **Locul 1: Top Recomandare**\n\n**{pariul_1}**\n\nȘansă: **{prob_1:.1f}%**")
    with col_b:
        st.info(f"🥈 **Locul 2: Alternativă Foarte Sigură**\n\n**{pariul_2}**\n\nȘansă: **{prob_2:.1f}%**")
    with col_c:
        st.warning(f"🥉 **Locul 3: Bilet de Siguranță**\n\n**{pariul_3}**\n\nȘansă: **{prob_3:.1f}%**")
        
    st.markdown("---")
    st.markdown("#### 📋 Clasament Complet al tuturor opțiunilor:")
    df_pariuri = pd.DataFrame(pariuri_sortate, columns=["Tip Pariu", "Probabilitate Matematică (%)"])
    df_pariuri["Probabilitate Matematică (%)"] = df_pariuri["Probabilitate Matematică (%)"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(df_pariuri, use_container_width=True)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1X2 Rezultat Final")
        st.write(f"• **1 (Victorie {echipa_gazda}):** {p1:.1f}%")
        st.write(f"• **X (Egal):** {px:.1f}%")
        st.write(f"• **2 (Victorie {echipa_oaspete}):** {p2:.1f}%")
        st.markdown("---")
        st.markdown("### Șansă Dublă")
        st.write(f"• **1X:** {p1+px:.1f}%")
        st.write(f"• **X2:** {px+p2:.1f}%")
        st.write(f"• **12:** {p1+p2:.1f}%")
        
    with col2:
        st.markdown("### Pauză sau Final (PsF)")
        st.write(f"• **PsF 1:** {psf1:.1f}%")
        st.write(f"• **PsF X:** {psfx:.1f}%")
        st.write(f"• **PsF 2:** {psf2:.1f}%")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Total Goluri (Meci)")
        st.write(f"• **Peste 1.5 Goluri:** {p_over15:.1f}% | **Sub 1.5:** {p_under15:.1f}%")
        st.write(f"• **Peste 2.5 Goluri:** {p_over25:.1f}% | **Sub 2.5:** {p_under25:.1f}%")
        st.write(f"• **Peste 3.5 Goluri:** {p_over35:.1f}% | **Sub 3.5:** {p_under35:.1f}%")
    with col2:
        st.markdown("### Intervale Goluri")
        st.write(f"• **1-3 Goluri în meci:** {p_g13:.1f}%")
        st.write(f"• **2-4 Goluri în meci:** {p_g24:.1f}%")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Rezultat Repriza 1")
        st.write(f"• **1 R1:** {p1_r1:.1f}%")
        st.write(f"• **X R1:** {px_r1:.1f}%")
        st.write(f"• **2 R1:** {p2_r1:.1f}%")
    with col2:
        st.markdown("### Total Goluri Repriza 1")
        st.write(f"• **Peste 0.5 Goluri R1:** {100 - mat_r1[0,0]*100:.1f}%")
        st.write(f"• **Peste 1.5 Goluri R1:** {100 - (mat_r1[0,0]+mat_r1[1,0]+mat_r1[0,1])*100:.1f}%")

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Ambele Marchează (Meci)")
        st.write(f"• **GG (Da):** {p_gg:.1f}%")
        st.write(f"• **NGG (Nu):** {100 - p_gg:.1f}%")
    with col2:
        st.markdown("### Ambele Marchează Reprize")
        st.write(f"• **GG Repriza 1:** {p_gg_r1:.1f}%")
        st.write(f"• **GG În Ambele Reprize:** {(p_gg_r1 * 0.4):.1f}%")

with tab5:
    st.markdown("### Pariuri pe Cartonașe (Superbet)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• **Peste 3.5 Cartonașe:** {p_cart_over35:.1f}%")
        st.write(f"• **Peste 4.5 Cartonașe:** {p_cart_over45:.1f}%")
    with col2:
        st.write(f"• **Sub 3.5 Cartonașe:** {100 - p_cart_over35:.1f}%")
        st.write(f"• **Sub 4.5 Cartonașe:** {100 - p_cart_over45:.1f}%")

with tab6:
    st.markdown("### Pariuri pe Cornere (Superbet)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• **Peste 8.5 Cornere:** {p_corn_over85:.1f}%")
        st.write(f"• **Peste 9.5 Cornere:** {p_corn_over95:.1f}%")
    with col2:
        st.write(f"• **Sub 8.5 Cornere:** {100 - p_corn_over85:.1f}%")
        st.write(f"• **Sub 9.5 Cornere:** {100 - p_corn_over95:.1f}%")
