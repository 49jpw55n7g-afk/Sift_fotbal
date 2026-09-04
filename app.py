import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

st.set_page_config(page_title="Predictor Automatizat Superbet", page_icon="⚽", layout="wide")

st.title("⚽ PREDICTOR ADVANCED - XG, GOLURI, CORNERE & CARTONAȘE FILTRATE")

st.sidebar.header("⚙️ Conectare API")
api_key = st.sidebar.text_input("Cheie API Football-Data.org", value="20505c2f8aaa48e58a6c4764d0664e7f", type="password")

# =============================================================================
# MODUL DE FILTRARE STRICTĂ A RISCULUI (GOLURI, GG, CORNERE, CARTONAȘE)
# =============================================================================

def filtreaza_piata_goluri(tip_pariu: str, linie: float, xg_gazde: float, xg_oaspeti: float, meciuri_gazde: list, meciuri_oaspeti: list):
    xg_total = xg_gazde + xg_oaspeti
    tip_pariu = tip_pariu.upper()
    
    if tip_pariu == 'SUB':
        marja_siguranta = 0.8 if linie <= 2.5 else 1.2
        prag_max_xg = linie - marja_siguranta
        
        if xg_total > prag_max_xg:
            return False, f"RESPINS: xG Total ({xg_total:.2f}) peste pragul de siguranță ({prag_max_xg:.2f}) pentru Sub {linie}"
        
        if xg_gazde > (linie / 2) + 0.2 or xg_oaspeti > (linie / 2) + 0.2:
            return False, f"RESPINS: Atac prea puternic la una din echipe pentru Sub {linie}"
            
        std_g = np.std(meciuri_gazde) if meciuri_gazde else 0
        std_o = np.std(meciuri_oaspeti) if meciuri_oaspeti else 0
        if std_g > 1.2 or std_o > 1.2:
            return False, "RESPINS: Fluctuații mari de scoruri în meciurile recente"

        return True, f"APROBAT: Risc scăzut pentru Sub {linie} goluri"

    elif tip_pariu == 'PESTE':
        prag_min_xg = linie + 0.3
        
        if xg_total < prag_min_xg:
            return False, f"RESPINS: xG Total ({xg_total:.2f}) sub pragul minim ({prag_min_xg:.2f}) pentru Peste {linie}"
        
        min_xg_indiv = 0.80 if linie <= 2.5 else 1.10
        if xg_gazde < min_xg_indiv or xg_oaspeti < min_xg_indiv:
            return False, f"RESPINS: Una dintre echipe are xG individual sub {min_xg_indiv:.2f}"
            
        peste_g = sum(1 for g in meciuri_gazde if g > linie)
        peste_o = sum(1 for g in meciuri_oaspeti if g > linie)
        if peste_g < 3 or peste_o < 3:
            return False, f"RESPINS: Rata meciurilor recente cu Peste {linie} este sub 60%"

        return True, f"APROBAT: Potențial ridicat pentru Peste {linie} goluri"

    return False, "Tip pariu invalid"


def filtreaza_piata_gg(tip_pariu: str, xg_gazde: float, xg_oaspeti: float, 
                       meciuri_gazde: list, meciuri_oaspeti: list,
                       clean_sheets_gazde: int = 0, clean_sheets_oaspeti: int = 0):
    tip_pariu = tip_pariu.upper()
    
    if tip_pariu == 'GG':
        if xg_gazde < 1.15 or xg_oaspeti < 1.15:
            return False, "RESPINS: xG scăzut la una din echipe (Minim 1.15 necesar per echipă)"
        
        gg_g = sum(1 for g in meciuri_gazde if g > 1)
        gg_o = sum(1 for g in meciuri_oaspeti if g > 1)
        
        if gg_g < 3 or gg_o < 3:
            return False, "RESPINS: Consistență scăzută în marcarea golurilor recente (<60%)"
            
        if clean_sheets_gazde >= 3 or clean_sheets_oaspeti >= 3:
            return False, "RESPINS: Apărare prea solidă la una din echipe (3+ Clean Sheets)"

        return True, "APROBAT: Meci excelent pentru GG (Ambele marchează)"

    elif tip_pariu == 'NGG':
        if xg_gazde >= 1.10 and xg_oaspeti >= 1.10:
            return False, "RESPINS: Ambele echipe au potențial ofensiv mare (xG > 1.10)"
            
        return True, "APROBAT: Șansă mare de NGG"

    return False, "Tip pariu invalid"


def filtreaza_piata_cornere(tip_pariu: str, linie: float, medie_cornere: float, istoric_cornere: list):
    """
    Filtru strict pentru piața de Cornere.
    """
    tip_pariu = tip_pariu.upper()
    std_cornere = np.std(istoric_cornere) if istoric_cornere else 0
    
    if tip_pariu == 'PESTE':
        prag_min = linie + 1.2  # Ex: Pentru Peste 8.5, media trebuie să fie cel puțin 9.7
        if medie_cornere < prag_min:
            return False, f"RESPINS: Medie cornere ({medie_cornere:.1f}) sub pragul minim de siguranță ({prag_min:.1f})"
        
        if std_cornere > 2.8:
            return False, f"RESPINS: Volatilitate mare pe cornere (std: {std_cornere:.2f})"
            
        meciuri_peste = sum(1 for c in istoric_cornere if c > linie)
        if meciuri_peste < 3:
            return False, f"RESPINS: Sub 60% din ultimele meciuri au avut peste {linie} cornere"
            
        return True, f"APROBAT: Meci cu potențial mare de Peste {linie} Cornere"

    elif tip_pariu == 'SUB':
        prag_max = linie - 1.2
        if medie_cornere > prag_max:
            return False, f"RESPINS: Medie cornere ({medie_cornere:.1f}) depășește pragul maxim ({prag_max:.1f})"
            
        return True, f"APROBAT: Risc scăzut pentru Sub {linie} Cornere"

    return False, "Tip pariu invalid"


def filtreaza_piata_cartonase(tip_pariu: str, linie: float, medie_cartonase: float, meci_derby: bool = False):
    """
    Filtru strict pentru piața de Cartonașe.
    """
    tip_pariu = tip_pariu.upper()
    
    if tip_pariu == 'PESTE':
        prag_min = linie + 0.8  # Ex: Pentru Peste 3.5, media trebuie să fie cel puțin 4.3
        if medie_cartonase < prag_min and not meci_derby:
            return False, f"RESPINS: Medie cartonașe ({medie_cartonase:.1f}) sub pragul minim ({prag_min:.1f})"
            
        return True, f"APROBAT: Tensiune ridicată / Meci propice pentru Peste {linie} Cartonașe"

    elif tip_pariu == 'SUB':
        prag_max = linie - 0.8
        if medie_cartonase > prag_max or meci_derby:
            return False, f"RESPINS: Meci cu risc ridicat de durități (Medie: {medie_cartonase:.1f})"
            
        return True, f"APROBAT: Joc curat anticipat / Sub {linie} Cartonașe"

    return False, "Tip pariu invalid"

# =============================================================================
# EXTRAGERE DATE API & LOGICĂ PRINCIPALĂ
# =============================================================================

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
def get_team_advanced_stats(team_id, api_key):
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=8"
    try:
        res = requests.get(url, headers={"X-Auth-Token": api_key})
        if res.status_code == 200:
            data = res.json().get("matches", [])
            if not data:
                return 1.2, 1.2, 1.2, 1.2, [1, 2, 1, 2, 1], 0
            
            scored_list, conceded_list, total_goals_list = [], [], []
            clean_sheets = 0
            
            for m in data:
                is_home = (m['homeTeam']['id'] == team_id)
                scored = m['score']['fullTime']['home'] if is_home else m['score']['fullTime']['away']
                conceded = m['score']['fullTime']['away'] if is_home else m['score']['fullTime']['home']
                if scored is not None and conceded is not None:
                    scored_list.append(scored)
                    conceded_list.append(conceded)
                    total_goals_list.append(scored + conceded)
                    if conceded == 0:
                        clean_sheets += 1
            
            avg_scored = np.mean(scored_list) if scored_list else 1.2
            avg_conceded = np.mean(conceded_list) if conceded_list else 1.2
            last_match_scored = scored_list[-1] if scored_list else avg_scored
            last_match_conceded = conceded_list[-1] if conceded_list else avg_conceded
            
            return (round(avg_scored, 2), round(avg_conceded, 2), 
                    float(last_match_scored), float(last_match_conceded), 
                    total_goals_list[-5:], clean_sheets)
        return 1.2, 1.2, 1.2, 1.2, [1, 2, 1, 2, 1], 0
    except:
        return 1.2, 1.2, 1.2, 1.2, [1, 2, 1, 2, 1], 0

matches = fetch_matches(api_key)

if matches:
    match_options = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['competition']['name']})": m for m in matches}
    if "selected_match_key" not in st.session_state:
        st.session_state.selected_match_key = list(match_options.keys())[0]

    selected_match_name = st.selectbox("Alege Meciul Zilei", list(match_options.keys()), key="selected_match_key")
    selected_match = match_options[selected_match_name]
    echipa_gazda = selected_match['homeTeam']['name']
    echipa_oaspete = selected_match['awayTeam']['name']
    
    with st.spinner("Se analizează meciurile recente și indicatorii statistici..."):
        h_avg_s, h_avg_c, h_last_s, h_last_c, h_recent_goals, h_cs = get_team_advanced_stats(selected_match['homeTeam']['id'], api_key)
        a_avg_s, a_avg_c, a_last_s, a_last_c, a_recent_goals, a_cs = get_team_advanced_stats(selected_match['awayTeam']['id'], api_key)
else:
    echipa_gazda, echipa_oaspete = "Arsenal", "Chelsea"
    h_avg_s, h_avg_c, h_last_s, h_last_c, h_recent_goals, h_cs = 1.70, 1.10, 2.0, 1.0, [2, 3, 1, 4, 2], 2
    a_avg_s, a_avg_c, a_last_s, a_last_c, a_recent_goals, a_cs = 1.30, 1.40, 1.0, 2.0, [1, 2, 2, 0, 3], 1

# SIDEBAR CONFIGURĂRI
st.sidebar.subheader("🟨 Cartonașe & 🚩 Cornere")
medie_cartonase = st.sidebar.number_input("Medie Cartonașe / Meci", value=4.5, step=0.5, key="cart")
este_derby = st.sidebar.checkbox("Meci de mare rivalitate / Derby (Cartonașe +)", value=False)

medie_cornere = st.sidebar.number_input("Medie Cornere / Meci", value=9.8, step=0.5, key="corn")
istoric_cornere_meciuri = [10, 8, 11, 9, 12]  # Istoric simulare meciuri recente

# CALCUL XG
exp_g_home = round((h_avg_s + a_avg_c) / 2, 2)
exp_g_away = round((a_avg_s + h_avg_c) / 2, 2)

# POISSON CALCUL
max_goals = 6
mat_full = np.zeros((max_goals, max_goals))
for i in range(max_goals):
    for j in range(max_goals):
        mat_full[i, j] = poisson.pmf(i, exp_g_home) * poisson.pmf(j, exp_g_away)

p_under35 = float(np.sum([mat_full[i, j] for i in range(4) for j in range(4) if i + j < 4]) * 100)
p_over25 = 100 - float(np.sum([mat_full[i, j] for i in range(3) for j in range(3) if i + j < 3]) * 100)
p_gg = float((1 - (np.sum(mat_full[0, :]) + np.sum(mat_full[:, 0]) - mat_full[0,0])) * 100)

p_corn_over85 = (1 - poisson.cdf(8, medie_cornere)) * 100
p_cart_over35 = (1 - poisson.cdf(3, medie_cartonase)) * 100

# FILTRE RULATE
eval_u35, msg_u35 = filtreaza_piata_goluri('SUB', 3.5, exp_g_home, exp_g_away, h_recent_goals, a_recent_goals)
eval_o25, msg_o25 = filtreaza_piata_goluri('PESTE', 2.5, exp_g_home, exp_g_away, h_recent_goals, a_recent_goals)
eval_gg, msg_gg = filtreaza_piata_gg('GG', exp_g_home, exp_g_away, h_recent_goals, a_recent_goals, h_cs, a_cs)

eval_corn, msg_corn = filtreaza_piata_cornere('PESTE', 8.5, medie_cornere, istoric_cornere_meciuri)
eval_cart, msg_cart = filtreaza_piata_cartonase('PESTE', 3.5, medie_cartonase, este_derby)

toate_pariurile = {
    "Goluri: Sub 3.5 Goluri": (p_under35, eval_u35, msg_u35),
    "Goluri: Peste 2.5 Goluri": (p_over25, eval_o25, msg_o25),
    "Goluri: Ambele Marchează (GG)": (p_gg, eval_gg, msg_gg),
    "Cornere: Peste 8.5 Cornere": (p_corn_over85, eval_corn, msg_corn),
    "Cartonașe: Peste 3.5 Cartonașe": (p_cart_over35, eval_cart, msg_cart)
}

pariuri_filtrate = [(k, v[0], v[1], v[2]) for k, v in toate_pariurile.items() if v[1] == True]
pariuri_filtrate_sortate = sorted(pariuri_filtrate, key=lambda x: x[1], reverse=True)

# DISPLAY STREAMLIT
st.subheader(f"🏟️ {echipa_gazda} vs {echipa_oaspete}")

tab_top, tab_filtre = st.tabs(["🏆 Recomandări Aprobate", "🛡️ Detalii Filtre Cornere & Cartonașe"])

with tab_top:
    st.markdown("### 🚀 Selecții Aprobate (Risc Redus)")
    
    if pariuri_filtrate_sortate:
        data_table = []
        for item in pariuri_filtrate_sortate:
            data_table.append((item[0], f"{item[1]:.1f}%", f"{100/item[1]:.2f}", item[3]))
            
        df_aprobate = pd.DataFrame(data_table, columns=["Tip Pariu", "Probabilitate", "Cotă Estimată", "Status Filtru"])
        st.dataframe(df_aprobate, use_container_width=True)
    else:
        st.warning("⚠️ Nicio selecție nu a trecut filtrele stricte pentru acest meci.")

with tab_filtre:
    st.markdown("### 🚩 Verificare Filtru Cornere (Peste 8.5)")
    if eval_corn:
        st.success(f"✅ {msg_corn}")
    else:
        st.error(f"❌ {msg_corn}")
        
    st.markdown("### 🟨 Verificare Filtru Cartonașe (Peste 3.5)")
    if eval_cart:
        st.success(f"✅ {msg_cart}")
    else:
        st.error(f"❌ {msg_cart}")
