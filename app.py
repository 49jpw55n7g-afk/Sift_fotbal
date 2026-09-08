import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import poisson
from datetime import datetime, timezone, timedelta

# ============================================================
# 1. CONFIGURARE PAGINĂ STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Quantum Analytics Engine - Year 3000 Edition",
    page_icon="⚡",
    layout="wide"
)

MAX_GOALS = 10
DECAY_DAYS = 180
SHRINKAGE_GAMES = 8
DEFAULT_RHO = -0.10

COMPETITIONS = {
    "PL": "Premier League (Anglia)",
    "PD": "La Liga (Spania)",
    "SA": "Serie A (Italia)",
    "BL1": "Bundesliga (Germania)",
    "FL1": "Ligue 1 (Franța)",
    "RO1": "SuperLiga (România)",
    "CL": "UEFA Champions League",
    "EL": "UEFA Europa League",
    "ECL": "UEFA Conference League",
    "FAC": "FA Cup",
    "CDR": "Copa del Rey",
    "DFB": "DFB-Pokal",
    "CI": "Coppa Italia",
    "CDF": "Coupe de France",
    "CR": "Cupa României"
}

# Persistent API Key State
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

# ============================================================
# 2. UTILS & MATHEMATICS ENGINE
# ============================================================

def clamp(x, low, high):
    return max(low, min(high, float(x)))

def safe_probability(p):
    return clamp(p, 1e-6, 1.0 - 1e-6)

def fair_odds(prob):
    return 1.0 / safe_probability(prob)

def get_temporal_weight(match_date, ref_date=None):
    if ref_date is None:
        ref_date = datetime.now(timezone.utc).replace(tzinfo=None)
    
    if isinstance(match_date, str):
        try:
            match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return 0.5
    elif isinstance(match_date, datetime):
        match_date = match_date.replace(tzinfo=None)

    age_days = max(0, (ref_date - match_date).days)
    return clamp(math.exp(-age_days / DECAY_DAYS), 0.05, 1.0)

def build_real_team_database(matches):
    teams = {}
    home_goals, away_goals = [], []

    for m in matches:
        if not m.get("completed"):
            continue

        sh, sa = m.get("score_home"), m.get("score_away")
        home, away = m.get("home"), m.get("away")

        if None in (sh, sa, home, away):
            continue

        sh, sa = float(sh), float(sa)
        weight = get_temporal_weight(m.get("date"))

        home_goals.append(sh)
        away_goals.append(sa)

        for team, venue, gf, ga in [(home, "home", sh, sa), (away, "away", sa, sh)]:
            if team not in teams:
                teams[team] = []
            teams[team].append({"venue": venue, "goals_for": gf, "goals_against": ga, "weight": weight})

    avg_home = float(np.mean(home_goals)) if home_goals else 1.45
    avg_away = float(np.mean(away_goals)) if away_goals else 1.15
    return teams, avg_home, avg_away

def calculate_team_strength(team, venue, database, league_avg):
    matches = [m for m in database.get(team, []) if m["venue"] == venue]
    if not matches:
        return {"attack": 1.0, "defense": 1.0, "games": 0}

    gf = np.array([m["goals_for"] for m in matches])
    ga = np.array([m["goals_against"] for m in matches])
    weights = np.array([m["weight"] for m in matches])

    total_w = np.sum(weights)
    avg_gf = np.average(gf, weights=weights) if total_w > 0 else np.mean(gf)
    avg_ga = np.average(ga, weights=weights) if total_w > 0 else np.mean(ga)

    games = len(matches)
    shrink = games / (games + SHRINKAGE_GAMES)

    adj_gf = (shrink * avg_gf) + ((1.0 - shrink) * league_avg)
    adj_ga = (shrink * avg_ga) + ((1.0 - shrink) * league_avg)

    return {"attack": float(adj_gf / league_avg), "defense": float(adj_ga / league_avg), "games": games}

def calculate_recent_form(team, database, last_n=5):
    matches = database.get(team, [])
    if not matches:
        return 1.0

    sorted_m = sorted(matches, key=lambda x: x["weight"], reverse=True)[:last_n]
    points = [3 if m["goals_for"] > m["goals_against"] else (1 if m["goals_for"] == m["goals_against"] else 0) for m in sorted_m]
    avg_p = np.mean(points) if points else 1.36
    return float(0.80 + (avg_p / 3.0) * 0.40)

def dixon_coles_tau(h, a, l_home, l_away, rho):
    if h == 0 and a == 0: return 1.0 - (l_home * l_away * rho)
    elif h == 0 and a == 1: return 1.0 + (l_home * rho)
    elif h == 1 and a == 0: return 1.0 + (l_away * rho)
    elif h == 1 and a == 1: return 1.0 - rho
    return 1.0

def build_dixon_coles_matrix(l_home, l_away, rho=DEFAULT_RHO, max_goals=MAX_GOALS):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base_p = poisson.pmf(h, l_home) * poisson.pmf(a, l_away)
            tau = dixon_coles_tau(h, a, l_home, l_away, rho)
            matrix[h, a] = base_p * tau

    total = np.sum(matrix)
    return matrix / total if total > 0 else matrix

def calculate_expected_goals(home_team, away_team, database, avg_home, avg_away):
    h_str = calculate_team_strength(home_team, "home", database, avg_home)
    a_str = calculate_team_strength(away_team, "away", database, avg_away)

    l_home = avg_home * h_str["attack"] * a_str["defense"]
    l_away = avg_away * a_str["attack"] * h_str["defense"]

    l_home *= (0.85 + 0.15 * calculate_recent_form(home_team, database))
    l_away *= (0.85 + 0.15 * calculate_recent_form(away_team, database))

    return clamp(l_home, 0.2, 4.5), clamp(l_away, 0.2, 4.5)

def extract_all_markets(matrix):
    size = matrix.shape[0]
    p_1 = float(np.sum(np.tril(matrix, -1)))
    p_x = float(np.sum(np.diag(matrix)))
    p_2 = float(np.sum(np.triu(matrix, 1)))

    p_btts_yes = float(np.sum(matrix[1:, 1:]))
    
    markets = {
        "1 (Gazde)": p_1, "X (Egal)": p_x, "2 (Oaspeți)": p_2,
        "1X (Șansă Dublă)": p_1 + p_x, "X2 (Șansă Dublă)": p_x + p_2, "12 (Fără Egal)": p_1 + p_2,
        "GG (Ambele Marchează)": p_btts_yes, "NG (Nu Marchează Ambele)": 1.0 - p_btts_yes
    }

    for line in [1.5, 2.5, 3.5]:
        over_p = sum(matrix[h, a] for h in range(size) for a in range(size) if h + a > line)
        markets[f"Peste {line} Goluri"] = float(over_p)
        markets[f"Sub {line} Goluri"] = float(1.0 - over_p)

    markets["1X & Peste 1.5"] = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h >= a and (h + a) > 1.5))
    markets["X2 & Peste 1.5"] = float(sum(matrix[h, a] for h in range(size) for a in range(size) if a >= h and (h + a) > 1.5))

    return markets

# ============================================================
# 3. AUTO SYNC API ENGINE
# ============================================================

@st.cache_data(ttl=900)
def fetch_all_matches_auto(api_key):
    if not api_key or len(api_key.strip()) < 8:
        return [], []

    headers = {'X-Auth-Token': api_key.strip()}
    today_dt = datetime.now(timezone.utc)
    today_str = today_dt.strftime("%Y-%m-%d")
    past_60_str = (today_dt - timedelta(days=60)).strftime("%Y-%m-%d")

    upcoming = []
    historical = []

    for code, comp_name in COMPETITIONS.items():
        # Fetch today matches
        url_up = f"https://api.football-data.org/v4/competitions/{code}/matches?dateFrom={today_str}&dateTo={today_str}"
        try:
            res = requests.get(url_up, headers=headers, timeout=4)
            if res.status_code == 200:
                for m in res.json().get("matches", []):
                    upcoming.append({
                        "id": m.get("id"),
                        "league": comp_name,
                        "home": m.get("homeTeam", {}).get("name", "Gazde"),
                        "away": m.get("awayTeam", {}).get("name", "Oaspeți"),
                        "date": m.get("utcDate", today_str)[:10],
                        "status": m.get("status")
                    })
        except Exception:
            pass

        # Fetch historical data for machine learning
        url_hist = f"https://api.football-data.org/v4/competitions/{code}/matches?dateFrom={past_60_str}&dateTo={today_str}"
        try:
            res_h = requests.get(url_hist, headers=headers, timeout=4)
            if res_h.status_code == 200:
                for m in res_h.json().get("matches", []):
                    if m.get("status") == "FINISHED":
                        sc = m.get("score", {}).get("fullTime", {})
                        if sc.get("home") is not None and sc.get("away") is not None:
                            historical.append({
                                "home": m.get("homeTeam", {}).get("name"),
                                "away": m.get("awayTeam", {}).get("name"),
                                "score_home": sc.get("home"),
                                "score_away": sc.get("away"),
                                "completed": True,
                                "date": m.get("utcDate")[:10]
                            })
        except Exception:
            pass

    return historical, upcoming

# ============================================================
# 4. INTERFAȚĂ UTILIZATOR STREAMLIT
# ============================================================

st.title("⚡ Quantum Analytics Engine — Year 3000 Edition")
st.caption("🤖 Predictor Autonom & Generator Biletul Zilei cu Auto-Învățare")

input_key = st.sidebar.text_input("🔑 Cheie API Football-Data.org:", value=st.session_state["api_key"], type="password")

if input_key:
    st.session_state["api_key"] = input_key

if not st.session_state["api_key"]:
    st.info("👈 Introduceți Cheia API în bara laterală. Cheia va fi salvată automat pentru întreaga sesiune.")
else:
    with st.spinner("🔄 Se sincronizează meciurile de AZI și se actualizează modelul matematic..."):
        historical_matches, upcoming_matches = fetch_all_matches_auto(st.session_state["api_key"])

    db, avg_home, avg_away = build_real_team_database(historical_matches)
    st.sidebar.success(f"🧠 Bază de date actualizată: {len(historical_matches)} meciuri învățate!")

    if not upcoming_matches:
        st.warning("⚠️ Nu există meciuri programate pentru AZI în ligile selectate sau abonamentul API nu include ligile respective.")
    else:
        st.sidebar.header("🎯 Meciuri de Azi")
        
        match_map = {f"[{m['league']}] {m['home']} vs {m['away']}": m for m in upcoming_matches}
        selected_labels = st.sidebar.multiselect(
            "Alege meciurile de analizat:",
            options=list(match_map.keys()),
            default=list(match_map.keys())
        )

        selected_matches = [match_map[lbl] for lbl in selected_labels]

        analyzed_matches = []
        ticket_candidates = []

        for m in selected_matches:
            l_h, l_a = calculate_expected_goals(m["home"], m["away"], db, avg_home, avg_away)
            matrix = build_dixon_coles_matrix(l_h, l_a)
            mkts = extract_all_markets(matrix)

            # Găsește cea mai sigură opțiune pentru bilet
            best_pick = max(mkts.items(), key=lambda x: x[1])
            
            analyzed_matches.append({
                "match": m, "l_h": l_h, "l_a": l_a, "markets": mkts
            })

            ticket_candidates.append({
                "Competiție": m["league"],
                "Meci": f"{m['home']} vs {m['away']}",
                "Pronostic Optim": best_pick[0],
                "Nivel Încredere": f"{best_pick[1] * 100:.1f}%",
                "Cotă Estimată (Fair)": f"{fair_odds(best_pick[1]):.2f}",
                "raw_prob": best_pick[1],
                "raw_odds": fair_odds(best_pick[1])
            })

        # ============================================================
        # GENERARE AUTOMATĂ BILETUL ZILEI
        # ============================================================
        st.markdown("---")
        st.header("🎟️ BILETUL ZILEI AUTOMAT (Șansă Maximă de Reușită)")

        if ticket_candidates:
            # Sortăm după probabilitate și alegem top 3-4 cele mai sigure
            best_ticket_picks = sorted(ticket_candidates, key=lambda x: x["raw_prob"], reverse=True)[:4]
            
            total_odds = 1.0
            avg_conf = np.mean([p["raw_prob"] for p in best_ticket_picks]) * 100

            for p in best_ticket_picks:
                total_odds *= p["raw_odds"]

            df_ticket = pd.DataFrame(best_ticket_picks).drop(columns=["raw_prob", "raw_odds"])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df_ticket, use_container_width=True)
            with col2:
                st.metric("Cotă Totală Bilet", f"{total_odds:.2f}")
                st.metric("Încredere Algoritm", f"{avg_conf:.1f}%")
                st.caption("Pariurile sunt generate automat pe baza calculelor de densitate de probabilitate Dixon-Coles.")

        # ============================================================
        # ANALIZĂ DETALIATĂ MECI CU MECI
        # ============================================================
        st.markdown("---")
        st.header(f"📊 Analiză Detaliată & Toată Gama de Pariuri ({len(analyzed_matches)})")

        for item in analyzed_matches:
            m = item["match"]
            l_h, l_a = item["l_h"], item["l_a"]
            mkts = item["markets"]

            with st.expander(f"⚽ [{m['league']}] {m['home']} vs {m['away']} | xG: {l_h:.2f} - {l_a:.2f}"):
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**🏆 Rezultat Final & Șansă Dublă**")
                    data_res = [
                        {"Piață": k, "Probabilitate": f"{v*100:.1f}%", "Cotă Fair": f"{fair_odds(v):.2f}"}
                        for k, v in mkts.items() if k in ["1 (Gazde)", "X (Egal)", "2 (Oaspeți)", "1X (Șansă Dublă)", "X2 (Șansă Dublă)", "12 (Fără Egal)"]
                    ]
                    st.dataframe(pd.DataFrame(data_res), use_container_width=True)

                with c2:
                    st.markdown("**⚽ Goluri & Combo**")
                    data_goals = [
                        {"Piață": k, "Probabilitate": f"{v*100:.1f}%", "Cotă Fair": f"{fair_odds(v):.2f}"}
                        for k, v in mkts.items() if k not in ["1 (Gazde)", "X (Egal)", "2 (Oaspeți)", "1X (Șansă Dublă)", "X2 (Șansă Dublă)", "12 (Fără Egal)"]
                    ]
                    st.dataframe(pd.DataFrame(data_goals), use_container_width=True)
