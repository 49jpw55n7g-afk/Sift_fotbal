import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

st.set_page_config(page_title="Predictor Automizat Superbet", page_icon="⚽", layout="wide")

st.title("⚽ PREDICTOR AUTOMATIZAT - XG RECENT & VALUE BETS")

# Sidebar - Configurare API
st.sidebar.header("⚙️ Conectare API")
api_key = st.sidebar.text_input("Cheie API Football-Data.org", value="20505c2f8aaa48e58a6c4764d0664e7f", type="password")

@st.cache_data(ttl=3600)
def fetch_matches(api_key):
    url = "https://api.football-data.org/v4/matches"
    try:
        response = requests.get(url, headers={"X-Auth-Token": api_key})
        if response.status_code == 200:
            return response.json().get("matches", [])
        return []
    except:
        return []

@st.cache_data(ttl=3600)
def get_team_stats(team_id, api_key):
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=6"
    try:
        res = requests.get(url, headers={"X-Auth-Token": api_key})
        if res.status_code == 200:
            data = res.json().get("matches", [])
            goals_scored = 0
            goals_conceded = 0
            count = len(data)
            if count == 0:
                return 1.4, 1.2
            for m in data:
                if m['homeTeam']['id'] == team_id:
                    goals_scored += m['score']['fullTime']['home'] or 0
                    goals_conceded += m['score']['fullTime']['away'] or 0
                else:
                    goals_scored += m['score']['fullTime']['away'] or 0
                    goals_conceded += m['score']['fullTime']['home'] or 0
            return round(goals_scored / count, 2), round(goals_conceded / count, 2)
        return 1.4, 1.2
    except:
        return 1.4, 1.2

matches = fetch_matches(api_key)

if matches:
    match_options = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['competition']['name']})": m for m in matches}
    
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
    
    with st.spinner("Se analizează ultimele meciuri ale echipelor..."):
        h_attack, h_defense = get_team_stats(selected_match['homeTeam']['id'], api_key)
        a_attack, a_defense = get_team_stats(selected_match['awayTeam']['id'], api_key)
        
        calculated_xg_home = round((h_attack + a_defense) / 2, 2)
        calculated_xg_away = round((a_attack + h_defense) / 2, 2)
        
        odds_data = selected_match.get('odds', {})
        cota_1 = odds_data.get('homeWin', None)
        cota_x = odds_data.get('draw', None)
        cota_2 = odds_data.get('awayWin', None)

else:
    st.info("Introduceți manual datele meciului.")
    echipa_gazda = st.sidebar.text_input("Echipă Gazdă", "Arsenal")
    echipa_oaspete = st.sidebar.text_input("Echipă Oaspete", "Chelsea")
    calculated_xg_home, calculated_xg_away = 1.70, 1.20
    cota_1, cota_x, cota_2 = None, None, None

# Sidebar - Parametri
st.sidebar.subheader("📊 Parametri Calculați Automat")
exp_g_home = st.sidebar.number_input("xG Gazdă (Formă Recentă)", value=float(calculated_xg_home), step=0.1, key=f"xg_h_{echipa_gazda}")
exp_g_away = st.sidebar.number_input("xG Oaspete (Formă Recentă)", value=float(calculated_xg_away), step=0.1, key=f"xg_a_{echipa_oaspete}")

st.sidebar.subheader("🟨 Cartonașe & 🚩 Cornere")
medie_cartonase = st.sidebar.number_input("Medie Cartonașe / Meci", value=4.5, step=0.5, key="cart")
medie_cornere = st.sidebar.number_input("Medie Cornere / Meci", value=9.5, step=0.5, key="corn")

# ---------------------------------------------------------
# CALCUL MATEMATIC & POISSON
# ---------------------------------------------------------
max_goals = 6

def build_poisson_matrix(exp_h, exp_a):
    mat = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            mat[i, j] = poisson.pmf(i, exp_h) * poisson.pmf(j, exp_a)
    return mat

mat_full = build_poisson_matrix(exp_g_home, exp_g_away)

p1_raw = float(np.sum(np.tril(mat_full, -1)) * 100)
px_raw = float(np.sum(np.diag(mat_full)) * 100)
p2_raw = float(np.sum(np.triu(mat_full, 1)) * 100)

if cota_1 and cota_x and cota_2:
    margin = (1/cota_1) + (1/cota_x) + (1/cota_2)
    p1_market = (1 / cota_1 / margin) * 100
    px_market = (1 / cota_x / margin) * 100
    p2_market = (1 / cota_2 / margin) * 100
    p1, px, p2 = (p1_raw + p1_market) / 2, (px_raw + px_market) / 2, (p2_raw + p2_market) / 2
else:
    p1, px, p2 = p1_raw, px_raw, p2_raw

exp_r1_h, exp_r1_a = exp_g_home * 0.43, exp_g_away * 0.43
mat_r1 = build_poisson_matrix(exp_r1_h, exp_r1_a)
p1_r1 = float(np.sum(np.tril(mat_r1, -1)) * 100)
px_r1 = float(np.sum(np.diag(mat_r1)) * 100)
p2_r1 = float(np.sum(np.triu(mat_r1, 1)) * 100)

psf1 = min(100.0, p1 + p1_r1 - (p1 * p1_r1 / 100))
psfx = min(100.0, px + px_r1 - (px * px_r1 / 100))
psf2 = min(100.0, p2 + p2_r1 - (p2 * p2_r1 / 100))

p_under15 = float(np.sum([mat_full[i, j] for i in range(2) for j in range(2) if i + j < 2]) * 100)
p_under25 = float(np.sum([mat_full[i, j] for i in range(3) for j in range(3) if i + j < 3]) * 100)
p_under35 = float(np.sum([mat_full[i, j] for i in range(4) for j in range(4) if i + j < 4]) * 100)

p_over15, p_over25, p_over35 = 100 - p_under15, 100 - p_under25, 100 - p_under35
p_gg = float((1 - (np.sum(mat_full[0, :]) + np.sum(mat_full[:, 0]) - mat_full[0,0])) * 100)
p_gg_r1 = float((1 - (np.sum(mat_r1[0, :]) + np.sum(mat_r1[:, 0]) - mat_r1[0,0])) * 100)

p_g13 = float(np.sum([mat_full[i, j] for i in range(4) for j in range(4) if 1 <= i + j <= 3]) * 100)
p_g24 = float(np.sum([mat_full[i, j] for i in range(5) for j in range(5) if 2 <= i + j <= 4]) * 100)

p_cart_over35 = (1 - poisson.cdf(3, medie_cartonase)) * 100
p_corn_over85 = (1 - poisson.cdf(8, medie_cornere)) * 100

toate_pariurile = {
    f"Șansă Dublă: 1X ({echipa_gazda} / X)": p1 + px,
    f"Șansă Dublă: X2 (X / {echipa_oaspete})": px + p2,
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
# ALGORITM SELECȚIE VALUE BET (COTA MAI MARE + ȘANȘĂ MARE)
# ---------------------------------------------------------
# Căutăm opțiuni care au probabilitate de minim 55%, dar au cota estimată cât mai mare (reducem pragul de "siguranță orbească")
candidate_value_bets = []
for tip_pariu, prob in toate_pariurile.items():
    if 52.0 <= prob <= 75.0:  # Plaja ideală pentru cote mari dar realizabile (Cote de ~1.40 - 1.95)
        cota_estimata = round(100 / prob, 2)
        # Scor de valoare: echilibru între cotă și șansa reală
        score = prob * cota_estimata
        candidate_value_bets.append((tip_pariu, prob, cota_estimata, score))

candidate_value_bets.sort(key=lambda x: x[3], reverse=True)

best_value_bet = candidate_value_bets[0] if candidate_value_bets else (pariuri_sortate[0][0], pariuri_sortate[0][1], round(100/pariuri_sortate[0][1], 2), 0)

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
    st.markdown("### 🚀 PARIUL DE VALOARE (Cota Mare + Șansă Ridicată)")
    st.info(
        f"🎯 **Cea mai profitabilă selecție (Value Bet):**\n\n"
        f"👉 **{best_value_bet[0]}**\n\n"
        f"• Probabilitate Matematică: **{best_value_bet[1]:.1f}%**\n\n"
        f"• Cotă Fair Estimată la case: **{best_value_bet[2]}**"
    )

    st.markdown("---")
    st.markdown("### 🔥 Top Cele mai sigure opțiuni (Șanse maxime de reușită)")
    
    col_a, col_b, col_c = st.columns(3)
    pariul_1, prob_1 = pariuri_sortate[0]
    pariul_2, prob_2 = pariuri_sortate[1]
    pariul_3, prob_3 = pariuri_sortate[2]
    
    with col_a:
        st.success(f"🥇 **Locul 1: Cel mai sigur**\n\n**{pariul_1}**\n\nȘansă: **{prob_1:.1f}%** (Cotă ~{100/prob_1:.2f})")
    with col_b:
        st.info(f"🥈 **Locul 2: Alternativă Sigură**\n\n**{pariul_2}**\n\nȘansă: **{prob_2:.1f}%** (Cotă ~{100/prob_2:.2f})")
    with col_c:
        st.warning(f"🥉 **Locul 3: Bilet de Siguranță**\n\n**{pariul_3}**\n\nȘansă: **{prob_3:.1f}%** (Cotă ~{100/prob_3:.2f})")
        
    st.markdown("---")
    st.markdown("#### 📋 Clasament Complet cu Cote Estimate:")
    
    df_pariuri = pd.DataFrame(
        [(item[0], f"{item[1]:.1f}%", f"{100/item[1]:.2f}") for item[0], item[1] in pariuri_sortate],
        columns=["Tip Pariu", "Probabilitate Matematică (%)", "Cotă Estimată Fair"]
    )
    st.dataframe(df_pariuri, use_container_width=True)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1X2 Rezultat Final")
        st.write(f"• **1 (Victorie {echipa_gazda}):** {p1:.1f}% (Cotă ~{100/p1 if p1>0 else 0:.2f})")
        st.write(f"• **X (Egal):** {px:.1f}% (Cotă ~{100/px if px>0 else 0:.2f})")
        st.write(f"• **2 (Victorie {echipa_oaspete}):** {p2:.1f}% (Cotă ~{100/p2 if p2>0 else 0:.2f})")
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
        st.write(f"• **Peste 3.5 Goluri:** {p_under35:.1f}% | **Sub 3.5:** {p_under35:.1f}%")
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
    with col2:
        st.write(f"• **Sub 3.5 Cartonașe:** {100 - p_cart_over35:.1f}%")

with tab6:
    st.markdown("### Pariuri pe Cornere (Superbet)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• **Peste 8.5 Cornere:** {p_corn_over85:.1f}%")
    with col2:
        st.write(f"• **Sub 8.5 Cornere:** {100 - p_corn_over85:.1f}%")
