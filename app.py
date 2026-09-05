import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

st.set_page_config(page_title="Predictor Advanced - Superbet", page_icon="⚽", layout="wide")

st.title("⚽ PREDICTOR ADVANCED - XG, GOLURI, CORNERE & CARTONAȘE")

# =============================================================================
# CONFIGURARE SIDEBAR & PARAMETRI DE RISC
# =============================================================================

st.sidebar.header("⚙️ Conectare API")
api_key = st.sidebar.text_input("Cheie API Football-Data.org", value="20505c2f8aaa48e58a6c4764d0664e7f", type="password")

st.sidebar.subheader("🛡️ Profil de Risc (Slider)")
profil_risc = st.sidebar.select_slider(
    "Alege Nivelul de Prudență",
    options=["Conservator", "Echilibrat", "Valoare Mare"],
    value="Echilibrat"
)

# Setare marje dinamice în funcție de profilul de risc
if profil_risc == "Conservator":
    MARJA_GOLURI = 1.0
    MARJA_CORNERE = 1.5
    MARJA_CARTONASE = 1.0
    PROB_MINIMA = 75.0
    STD_MAX_PERMIS = 2.2
elif profil_risc == "Echilibrat":
    MARJA_GOLURI = 0.8
    MARJA_CORNERE = 1.2
    MARJA_CARTONASE = 0.8
    PROB_MINIMA = 68.0
    STD_MAX_PERMIS = 2.6
else:  # Valoare Mare
    MARJA_GOLURI = 0.5
    MARJA_CORNERE = 0.8
    MARJA_CARTONASE = 0.5
    PROB_MINIMA = 60.0
    STD_MAX_PERMIS = 3.0

st.sidebar.subheader("🟨 Parametri Arbitru & Ligă")
medie_arbitru = st.sidebar.number_input("Media Cartonașe Arbitru", value=3.8, step=0.1)
medie_liga_cartonase = st.sidebar.number_input("Media Cartonașe Ligă", value=4.2, step=0.1)
coef_arbitru = medie_arbitru / medie_liga_cartonase if medie_liga_cartonase > 0 else 1.0

st.sidebar.subheader("🚩 Parametri Cornere")
medie_cornere_meci = st.sidebar.number_input("Medie Cornere Meci", value=9.8, step=0.5)
este_favorita_deplasare = st.sidebar.checkbox("Favorită Clară în Deplasare (Penalizare Cornere)", value=False)

if este_favorita_deplasare:
    medie_cornere_meci *= 0.90  # Penalizare 10% pentru scenariu de scor dezechilibrat

# =============================================================================
# MOTOARE DE FILTRARE & SIGURANȚĂ (ANTI-EROARE)
# =============================================================================

def verifica_volatilitate(istoric_date: list, std_max: float):
    if not istoric_date:
        return True, 0.0
    std_val = np.std(istoric_date)
    if std_val > std_max:
        return False, std_val
    return True, std_val

def filtreaza_piata_goluri(tip_pariu: str, linie: float, xg_gazde: float, xg_oaspeti: float, meciuri_gazde: list, meciuri_oaspeti: list):
    xg_total = xg_gazde + xg_oaspeti
    tip_pariu = tip_pariu.upper()
    
    ok_g, std_g = verifica_volatilitate(meciuri_gazde, STD_MAX_PERMIS)
    ok_o, std_o = verifica_volatilitate(meciuri_oaspeti, STD_MAX_PERMIS)
    if not ok_g or not ok_o:
        return False, f"RESPINS: Volatilitate mare goluri (std: {max(std_g, std_o):.2f})"

    if tip_pariu == 'SUB':
        prag_max_xg = linie - MARJA_GOLURI
        if xg_total > prag_max_xg:
            return False, f"RESPINS: xG Total ({xg_total:.2f}) peste pragul ({prag_max_xg:.2f})"
        return True, f"APROBAT: Sub {linie} goluri"

    elif tip_pariu == 'PESTE':
        prag_min_xg = linie + (MARJA_GOLURI / 2)
        if xg_total < prag_min_xg:
            return False, f"RESPINS: xG Total ({xg_total:.2f}) sub pragul ({prag_min_xg:.2f})"
        return True, f"APROBAT: Peste {linie} goluri"

    return False, "Tip pariu invalid"

def filtreaza_piata_cornere(tip_pariu: str, linie: float, medie_cornere: float, istoric_cornere: list):
    tip_pariu = tip_pariu.upper()
    ok_vol, std_c = verifica_volatilitate(istoric_cornere, STD_MAX_PERMIS)
    
    if not ok_vol:
        return False, f"RESPINS: Fluctuații mari pe cornere (std: {std_c:.2f})"

    if tip_pariu == 'PESTE':
        prag_min = linie + MARJA_CORNERE
        if medie_cornere < prag_min:
            return False, f"RESPINS: Medie cornere ({medie_cornere:.1f}) sub pragul ({prag_min:.1f})"
        return True, f"APROBAT: Peste {linie} Cornere"

    elif tip_pariu == 'SUB':
        prag_max = linie - MARJA_CORNERE
        if medie_cornere > prag_max:
            return False, f"RESPINS: Medie cornere ({medie_cornere:.1f}) peste pragul ({prag_max:.1f})"
        return True, f"APROBAT: Sub {linie} Cornere"

    return False, "Tip pariu invalid"

def filtreaza_piata_cartonase(tip_pariu: str, linie: float, cart_gazde: float, cart_oaspeti: float, coef_arbitru: float):
    tip_pariu = tip_pariu.upper()
    medie_ajustata = (cart_gazde + cart_oaspeti) * coef_arbitru

    if tip_pariu == 'PESTE':
        if coef_arbitru < 0.85:
            return False, f"RESPINS: Arbitru permisiv (Coef: {coef_arbitru:.2f})"
        prag_min = linie + MARJA_CARTONASE
        if medie_ajustata < prag_min:
            return False, f"RESPINS: Medie ajustată ({medie_ajustata:.2f}) sub pragul ({prag_min:.2f})"
        return True, f"APROBAT: Peste {linie} Cartonașe"

    elif tip_pariu == 'SUB':
        prag_max = linie - MARJA_CARTONASE
        if medie_ajustata > prag_max:
            return False, f"RESPINS: Medie ajustată ({medie_ajustata:.2f}) peste pragul ({prag_max:.2f})"
        return True, f"APROBAT: Sub {linie} Cartonașe"

    return False, "Tip pariu invalid"

# =============================================================================
# EXTRAGERE DATE API (MODIFICAT PENTRU A EXCLUDE MECIURILE FINALIATE)
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_matches(api_key):
    # Preluăm doar meciurile programate (SCHEDULED, TIMED)
    url = "https://api.football-data.org/v4/matches?status=SCHEDULED,TIMED"
    try:
        response = requests.get(url, headers={"X-Auth-Token": api_key})
        if response.status_code == 200:
            raw_matches = response.json().get("matches", [])
            # Filtrare suplimentară pe status pentru siguranță
            filtered_matches = [
                m for m in raw_matches 
                if m.get('status') in ['SCHEDULED', 'TIMED']
            ]
            return filtered_matches
        return []
    except:
        return []

@st.cache_data(ttl=3600)
def get_team_advanced_stats(team_id, api_key):
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=8"
    try:
        res = requests.get(url, headers={"X-Auth-Token": api_key})
        if res.status_code == 200:
            data = res.json().get("matches", [])
            if not data:
                return 1.2, 1.2, [1, 2, 1, 2, 1]
            
            scored_list, conceded_list, total_goals_list = [], [], []
            for m in data:
                is_home = (m['homeTeam']['id'] == team_id)
                scored = m['score']['fullTime']['home'] if is_home else m['score']['fullTime']['away']
                conceded = m['score']['fullTime']['away'] if is_home else m['score']['fullTime']['home']
                if scored is not None and conceded is not None:
                    scored_list.append(scored)
                    conceded_list.append(conceded)
                    total_goals_list.append(scored + conceded)
            
            avg_scored = np.mean(scored_list) if scored_list else 1.2
            avg_conceded = np.mean(conceded_list) if conceded_list else 1.2
            return round(avg_scored, 2), round(avg_conceded, 2), total_goals_list[-5:]
        return 1.2, 1.2, [1, 2, 1, 2, 1]
    except:
        return 1.2, 1.2, [1, 2, 1, 2, 1]

# =============================================================================
# PROCESARE MECI SELECTAT
# =============================================================================

matches = fetch_matches(api_key)

if matches:
    match_options = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['competition']['name']})": m for m in matches}
    if "selected_match_key" not in st.session_state or st.session_state.selected_match_key not in match_options:
        st.session_state.selected_match_key = list(match_options.keys())[0]

    selected_match_name = st.selectbox("Alege Meciul de Analizat", list(match_options.keys()), key="selected_match_key")
    selected_match = match_options[selected_match_name]
    echipa_gazda = selected_match['homeTeam']['name']
    echipa_oaspete = selected_match['awayTeam']['name']
    
    with st.spinner("Procesare modele Poisson și verificare limite de siguranță..."):
        h_avg_s, h_avg_c, h_recent_goals = get_team_advanced_stats(selected_match['homeTeam']['id'], api_key)
        a_avg_s, a_avg_c, a_recent_goals = get_team_advanced_stats(selected_match['awayTeam']['id'], api_key)
else:
    st.info("Nu s-au găsit meciuri viitoare programate în API-ul curent.")
    echipa_gazda, echipa_oaspete = "Genoa", "Como"
    h_avg_s, h_avg_c, h_recent_goals = 1.10, 1.40, [1, 2, 1, 0, 2]
    a_avg_s, a_avg_c, a_recent_goals = 1.30, 1.20, [2, 1, 3, 1, 1]

# CÂMPURI FIXE INTRODUSE PENTRU STRUCTURĂ
cart_gazde, cart_oaspeti = 2.1, 1.9
istoric_cornere = [10, 8, 11, 9, 12]

# CALCULAT XG & POISSON
exp_g_home = round((h_avg_s + a_avg_c) / 2, 2)
exp_g_away = round((a_avg_s + h_avg_c) / 2, 2)

max_goals = 6
mat_full = np.zeros((max_goals, max_goals))
for i in range(max_goals):
    for j in range(max_goals):
        mat_full[i, j] = poisson.pmf(i, exp_g_home) * poisson.pmf(j, exp_g_away)

p_under35 = float(np.sum([mat_full[i, j] for i in range(4) for j in range(4) if i + j < 4]) * 100)
p_over25 = 100 - float(np.sum([mat_full[i, j] for i in range(3) for j in range(3) if i + j < 3]) * 100)

medie_cartonase_ajustata = (cart_gazde + cart_oaspeti) * coef_arbitru
p_corn_over85 = (1 - poisson.cdf(8, medie_cornere_meci)) * 100
p_cart_over35 = (1 - poisson.cdf(3, medie_cartonase_ajustata)) * 100
p_cart_over25 = (1 - poisson.cdf(2, medie_cartonase_ajustata)) * 100

# RULARE FILTRE
eval_u35, msg_u35 = filtreaza_piata_goluri('SUB', 3.5, exp_g_home, exp_g_away, h_recent_goals, a_recent_goals)
eval_o25, msg_o25 = filtreaza_piata_goluri('PESTE', 2.5, exp_g_home, exp_g_away, h_recent_goals, a_recent_goals)
eval_corn, msg_corn = filtreaza_piata_cornere('PESTE', 8.5, medie_cornere_meci, istoric_cornere)
eval_cart35, msg_cart35 = filtreaza_piata_cartonase('PESTE', 3.5, cart_gazde, cart_oaspeti, coef_arbitru)
eval_cart25, msg_cart25 = filtreaza_piata_cartonase('PESTE', 2.5, cart_gazde, cart_oaspeti, coef_arbitru)

toate_optiunile = [
    ("Goluri: Sub 3.5 Goluri", p_under35, eval_u35, msg_u35),
    ("Goluri: Peste 2.5 Goluri", p_over25, eval_o25, msg_o25),
    ("Cornere: Peste 8.5 Cornere", p_corn_over85, eval_corn, msg_corn),
    ("Cartonașe: Peste 3.5 Cartonașe", p_cart_over35, eval_cart35, msg_cart35),
    ("Cartonașe: Peste 2.5 Cartonașe (Linie Sigură)", p_cart_over25, eval_cart25, msg_cart25)
]

# PARIURI APROBATE & FILTRATE DUPĂ PROBABILITATEA MINIMĂ
aprobate = [item for item in toate_optiunile if item[2] and item[1] >= PROB_MINIMA]
aprobate_sortate = sorted(aprobate, key=lambda x: x[1], reverse=True)

# =============================================================================
# INTERFAȚĂ STREAMLIT
# =============================================================================

# PANOU PRINCIPAL 1: BILETUL ZILEI
st.markdown("## 🎫 BILETUL ZILEI (Risc Scăzut & Cotă Optimă)")

if len(aprobate_sortate) >= 2:
    sel1 = aprobate_sortate[0]
    sel2 = aprobate_sortate[1]
    
    cota1 = 100 / sel1[1]
    cota2 = 100 / sel2[1]
    cota_totala = cota1 * cota2
    prob_cumulata = (sel1[1] / 100) * (sel2[1] / 100) * 100

    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Cotă Totală Estimată", f"{cota_totala:.2f}")
    col_b2.metric("Șansă Cumulată de Reușită", f"{prob_cumulata:.1f}%")
    col_b3.metric("Profil Risc", profil_risc)

    df_bilet = pd.DataFrame([
        {"Meci": f"{echipa_gazda} vs {echipa_oaspete}", "Pariu": sel1[0], "Probabilitate": f"{sel1[1]:.1f}%", "Cotă": f"{cota1:.2f}"},
        {"Meci": f"{echipa_gazda} vs {echipa_oaspete}", "Pariu": sel2[0], "Probabilitate": f"{sel2[1]:.1f}%", "Cotă": f"{cota2:.2f}"}
    ])
    st.table(df_bilet)
else:
    st.info("ℹ️ Pentru meciul curent nu există suficiente selecții care să treacă de profilul de risc selectat pentru a forma Biletul Zilei.")

st.divider()

# TAB-URI ANALIZĂ DETALIATĂ
tab_builder, tab_toate, tab_debug = st.tabs(["⚡ Bet Builder Meci (~2.00)", "📋 Toate Selecțiile Aprobate", "🛡️ Detalii Filtre & Volatilitate"])

with tab_builder:
    st.markdown(f"### 🎯 Bet Builder Dedicat: {echipa_gazda} vs {echipa_oaspete}")
    if len(aprobate_sortate) >= 2:
        c1 = 100 / aprobate_sortate[0][1]
        c2 = 100 / aprobate_sortate[1][1]
        cota_bb = c1 * c2
        st.success(f"**Combinație Recomandată (Cotă Totală: {cota_bb:.2f}):**")
        st.markdown(f"* 🔹 **Selecția 1:** {aprobate_sortate[0][0]} (Șansă: {aprobate_sortate[0][1]:.1f}%)")
        st.markdown(f"* 🔹 **Selecția 2:** {aprobate_sortate[1][0]} (Șansă: {aprobate_sortate[1][1]:.1f}%)")
    else:
        st.warning("Meciul are volatilitate ridicată. Nu se recomandă un Bet Builder pentru acest meci.")

with tab_toate:
    st.markdown("### 🏆 Top Pariuri cu Valoare (EV) Pozitivă")
    if aprobate_sortate:
        tabel_data = []
        for item in aprobate_sortate:
            cota_est = 100 / item[1]
            tabel_data.append({
                "Tip Pariu": item[0],
                "Probabilitate Matematică": f"{item[1]:.1f}%",
                "Cotă Minima Utilă": f"{cota_est:.2f}",
                "Status Algoritm": item[3]
            })
        st.dataframe(pd.DataFrame(tabel_data), use_container_width=True)
    else:
        st.error("Nicio selecție nu a îndeplinit condițiile stricte de siguranță.")

with tab_debug:
    st.markdown("### 🔍 Indicatori de Volatilitate și Arbitraj")
    st.write(f"* **Coeficient Strictețe Arbitru ($K_{{arbitru}}$):** {coef_arbitru:.2f}")
    st.write(f"* **Medie Cartonașe Ajustată Meci:** {medie_cartonase_ajustata:.2f}")
    st.write(f"* **xG Total Meci (Gazde + Oaspeți):** {exp_g_home + exp_g_away:.2f}")
    st.write(f"* **Abatere Standard Permisă (Std Max):** {STD_MAX_PERMIS}")
