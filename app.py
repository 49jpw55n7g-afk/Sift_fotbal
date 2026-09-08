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
    page_title="Quantum Analytics Engine",
    page_icon="⚡",
    layout="wide"
)

MAX_GOALS = 10
DECAY_DAYS = 180
SHRINKAGE_GAMES = 8

# ID UEFA Champions League = 2
COMPETITIONS = {
    2: "UEFA Champions League",
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga"
}

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

# ============================================================
# 2. HELPERE ȘI ENGINE MATEMATIC
# ============================================================

def clamp(x, low, high):
    return max(low, min(high, float(x)))

def safe_probability(p):
    return clamp(p, 1e-6, 1.0 - 1e-6)

def fair_odds(prob):
    return 1.0 / safe_probability(prob)

def get_temporal_weight(match_date):
    try:
        ref_date = datetime.now(timezone.utc).replace(tzinfo=None)
        if isinstance(match_date, str):
            match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00")).replace(tzinfo=None)
        age_days = max(0, (ref_date - match_date).days)
        return clamp(math.exp(-age_days / DECAY_DAYS), 0.05, 1.0)
    except Exception:
        return 0.5

def build_real_team_database(matches):
    teams = {}
    home_goals, away_goals = [], []

    for m in matches:
        sh, sa = m.get("score_home"), m.get("score_away")
        home, away = m.get("home"), m.get("away")

        if sh is None or sa is None or not home or not away:
            continue

        try:
            sh, sa = float(sh), float(sa)
            weight = get_temporal_weight(m.get("date"))

            home_goals.append(sh)
            away_goals.append(sa)

            for team, venue, gf, ga in [(home, "home", sh, sa), (away, "away", sa, sh)]:
                if team not in teams:
                    teams[team] = []
                teams[team].append({"venue": venue, "goals_for": gf, "goals_against": ga, "weight": weight})
        except Exception:
            continue

    avg_home = float(np.mean(home_goals)) if home_goals else 1.45
    avg_away = float(np.mean(away_goals)) if away_goals else 1.15
    return teams, avg_home, avg_away

def calculate_expected_goals(home_team, away_team, database, avg_home, avg_away):
    def get_str(team, venue, league_avg):
        matches = [m for m in database.get(team, []) if m["venue"] == venue]
        if not matches:
            return 1.0, 1.0
        gf = np.array([m["goals_for"] for m in matches])
        ga = np.array([m["goals_against"] for m in matches])
        weights = np.array([m["weight"] for m in matches])
        
        tw = np.sum(weights)
        avg_gf = np.average(gf, weights=weights) if tw > 0 else np.mean(gf)
        avg_ga = np.average(ga, weights=weights) if tw > 0 else np.mean(ga)
        
        shrink = len(matches) / (len(matches) + SHRINKAGE_GAMES)
        adj_gf = (shrink * avg_gf) + ((1.0 - shrink) * league_avg)
        adj_ga = (shrink * avg_ga) + ((1.0 - shrink) * league_avg)
        return float(adj_gf / league_avg), float(adj_ga / league_avg)

    h_att, h_def = get_str(home_team, "home", avg_home)
    a_att, a_def = get_str(away_team, "away", avg_away)

    l_home = avg_home * h_att * a_def
    l_away = avg_away * a_att * h_def

    return clamp(l_home, 0.2, 4.5), clamp(l_away, 0.2, 4.5)

def build_dixon_coles_matrix(l_home, l_away):
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            matrix[h, a] = poisson.pmf(h, l_home) * poisson.pmf(a, l_away)
    total = np.sum(matrix)
    return matrix / total if total > 0 else matrix

def extract_all_markets(matrix):
    size = matrix.shape[0]
    p_1 = float(np.sum(np.tril(matrix, -1)))
    p_x = float(np.sum(np.diag(matrix)))
    p_2 = float(np.sum(np.triu(matrix, 1)))
    p_btts = float(np.sum(matrix[1:, 1:]))

    markets = {
        "1 (Gazde)": p_1, "X (Egal)": p_x, "2 (Oaspeți)": p_2,
        "1X (Șansă Dublă)": p_1 + p_x, "X2 (Șansă Dublă)": p_x + p_2, "12 (Fără Egal)": p_1 + p_2,
        "GG (Ambele Marchează)": p_btts, "NG (Nu Marchează Ambele)": 1.0 - p_btts
    }

    for line in [1.5, 2.5, 3.5]:
        over_p = sum(matrix[h, a] for h in range(size) for a in range(size) if h + a > line)
        markets[f"Peste {line} Goluri"] = float(over_p)
        markets[f"Sub {line} Goluri"] = float(1.0 - over_p)

    return markets

# ============================================================
# 3. CONEXIUNE API-SPORTS (CU OBTINERE AUTOMATĂ SEZON & DIRECT AZI)
# ============================================================

@st.cache_data(ttl=300)
def fetch_data_api_sports(api_key, season_year):
    if not api_key or len(api_key.strip()) < 8:
        return [], []

    headers = {
        "x-rapidapi-key": api_key.strip(),
        "x-rapidapi-host": "v3.football.api-sports.io"
    }

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    upcoming, historical = [], []

    # Interogare DIRECTĂ pe data de AZI pentru TOATE meciurile din sistem (fără restricție de ligă inițială)
    try:
        url_today = f"https://v3.football.api-sports.io/fixtures?date={today_str}"
        res = requests.get(url_today, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json().get("response", [])
            for item in data:
                league_id = item.get("league", {}).get("id")
                comp_name = item.get("league", {}).get("name", "Competiție")
                
                upcoming.append({
                    "id": item.get("fixture", {}).get("id"),
                    "league_id": league_id,
                    "league": comp_name,
                    "home": item.get("teams", {}).get("home", {}).get("name", "Gazde"),
                    "away": item.get("teams", {}).get("away", {}).get("name", "Oaspeți"),
                    "date": str(item.get("fixture", {}).get("date", today_str))[:10]
                })
    except Exception:
        pass

    # Extragere istoric recent pentru Champions League (League ID = 2)
    try:
        url_hist = f"https://v3.football.api-sports.io/fixtures?league=2&season={season_year}&last=30"
        res_h = requests.get(url_hist, headers=headers, timeout=8)
        if res_h.status_code == 200:
            for item in res_h.json().get("response", []):
                status = item.get("fixture", {}).get("status", {}).get("short")
                if status in ["FT", "AET", "PEN"]:
                    historical.append({
                        "home": item.get("teams", {}).get("home", {}).get("name"),
                        "away": item.get("teams", {}).get("away", {}).get("name"),
                        "score_home": item.get("goals", {}).get("home"),
                        "score_away": item.get("goals", {}).get("away"),
                        "date": str(item.get("fixture", {}).get("date", ""))[:10]
                    })
    except Exception:
        pass

    return historical, upcoming

# ============================================================
# 4. INTERFAȚĂ UTILIZATOR
# ============================================================

st.title("⚡ Quantum Analytics Engine — UCL Special")

input_key = st.sidebar.text_input("🔑 Cheie API (RapidAPI / API-Sports):", value=st.session_state["api_key"], type="password")
season_input = st.sidebar.number_input("📅 Sezonul curent (Anul de start):", min_value=2023, max_value=2026, value=2026)

if input_key:
    st.session_state["api_key"] = input_key

if not st.session_state["api_key"]:
    st.info("👈 Introdu Cheia API în bara din stânga.")
else:
    with st.spinner("🔄 Se descarcă TOATE meciurile programate pentru AZI..."):
        historical_matches, upcoming_matches = fetch_data_api_sports(st.session_state["api_key"], season_input)

    db, avg_home, avg_away = build_real_team_database(historical_matches)
    
    if not upcoming_matches:
        st.error("❌ Nu s-au primit date de la API. Verifică dacă Cheia API este activă sau dacă ai depășit limita de apeluri gratuite (Rate Limit).")
    else:
        # Filtru opțional de competiții în sidebar
        leagues_found = list(set([m["league"] for m in upcoming_matches]))
        selected_league_filter = st.sidebar.multiselect("Filtrează după competiție:", options=leagues_found, default=[l for l in leagues_found if "Champions League" in l] or leagues_found)

        filtered_upcoming = [m for m in upcoming_matches if m["league"] in selected_league_filter]

        if not filtered_upcoming:
            st.warning("⚠️ Nu există meciuri pentru competiția selectată. Selectează altă competiție din bara laterală.")
        else:
            match_map = {f"[{m['league']}] {m['home']} vs {m['away']}": m for m in filtered_upcoming}
            selected_labels = st.multiselect("Meciuri selectate pentru calculul cotelor fair:", options=list(match_map.keys()), default=list(match_map.keys()))

            selected_matches = [match_map[lbl] for lbl in selected_labels]
            analyzed_matches, ticket_candidates = [], []

            for m in selected_matches:
                l_h, l_a = calculate_expected_goals(m["home"], m["away"], db, avg_home, avg_away)
                matrix = build_dixon_coles_matrix(l_h, l_a)
                mkts = extract_all_markets(matrix)

                best_pick = max(mkts.items(), key=lambda x: x[1])

                analyzed_matches.append({"match": m, "l_h": l_h, "l_a": l_a, "markets": mkts})
                ticket_candidates.append({
                    "Competiție": m["league"],
                    "Meci": f"{m['home']} vs {m['away']}",
                    "Pronostic Recommended": best_pick[0],
                    "Nivel Încredere": f"{best_pick[1]*100:.1f}%",
                    "Cotă Fair Estimată": f"{fair_odds(best_pick[1]):.2f}",
                    "prob": best_pick[1],
                    "odds": fair_odds(best_pick[1])
                })

            # Section: Biletul Zilei
            st.markdown("---")
            st.header("🎟️ BILETUL AUTOMAT UCL / AZI")

            if ticket_candidates:
                top_picks = sorted(ticket_candidates, key=lambda x: x["prob"], reverse=True)[:4]
                total_odds = float(np.prod([p["odds"] for p in top_picks]))
                avg_conf = float(np.mean([p["prob"] for p in top_picks])) * 100

                df_t = pd.DataFrame(top_picks).drop(columns=["prob", "odds"])

                c1, c2 = st.columns([3, 1])
                with c1:
                    st.dataframe(df_t, use_container_width=True)
                with c2:
                    st.metric("Cotă Totală Bilet", f"{total_odds:.2f}")
                    st.metric("Încredere Medie", f"{avg_conf:.1f}%")

            # Section: Detalii
            st.markdown("---")
            st.header("📊 Analiză Detaliată Meciuri")

            for item in analyzed_matches:
                m, l_h, l_a, mkts = item["match"], item["l_h"], item["l_a"], item["markets"]
                with st.expander(f"⚽ [{m['league']}] {m['home']} vs {m['away']} | xG: {l_h:.2f} - {l_a:.2f}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**1X2 & Șansă Dublă**")
                        st.dataframe(pd.DataFrame([{"Piață": k, "Probabilitate": f"{v*100:.1f}%", "Cotă Fair": f"{fair_odds(v):.2f}"} for k, v in mkts.items() if "Goluri" not in k and "GG" not in k and "NG" not in k]), use_container_width=True)
                    with c2:
                        st.markdown("**Goluri & Ambele Marchează**")
                        st.dataframe(pd.DataFrame([{"Piață": k, "Probabilitate": f"{v*100:.1f}%", "Cotă Fair": f"{fair_odds(v):.2f}"} for k, v in mkts.items() if "Goluri" in k or "GG" in k or "NG" in k]), use_container_width=True)
