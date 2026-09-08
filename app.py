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
    page_title="Quantum Analytics Engine — Year 3000 Edition",
    page_icon="⚡",
    layout="wide"
)

MAX_GOALS = 10
DECAY_DAYS = 180
SHRINKAGE_GAMES = 8

# ID-uri oficiale API-Football (v3.football.api-sports.io)
COMPETITIONS = {
    39: "Premier League (Anglia)",
    140: "La Liga (Spania)",
    135: "Serie A (Italia)",
    78: "Bundesliga (Germania)",
    61: "Ligue 1 (Franța)",
    283: "SuperLiga (România)",
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    848: "UEFA Conference League",
    45: "FA Cup",
    143: "Copa del Rey",
    81: "DFB-Pokal",
    137: "Coppa Italia",
    66: "Coupe de France",
    513: "Cupa României"
}

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

# ============================================================
# 2. ENGINE MATEMATIC & DIXON-COLES
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
        return {"attack": 1.0, "defense": 1.0}

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

    return {"attack": float(adj_gf / league_avg), "defense": float(adj_ga / league_avg)}

def calculate_expected_goals(home_team, away_team, database, avg_home, avg_away):
    h_str = calculate_team_strength(home_team, "home", database, avg_home)
    a_str = calculate_team_strength(away_team, "away", database, avg_away)

    l_home = avg_home * h_str["attack"] * a_str["defense"]
    l_away = avg_away * a_str["attack"] * h_str["defense"]

    return clamp(l_home, 0.2, 4.5), clamp(l_away, 0.2, 4.5)

def build_dixon_coles_matrix(l_home, l_away, max_goals=MAX_GOALS):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
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
# 3. SINCRONIZARE API-FOOTBALL V3 (API-SPORTS)
# ============================================================

@st.cache_data(ttl=900)
def fetch_data_api_sports(api_key):
    if not api_key or len(api_key.strip()) < 8:
        return [], []

    headers = {
        "x-rapidapi-key": api_key.strip(),
        "x-rapidapi-host": "v3.football.api-sports.io"
    }

    today_dt = datetime.now(timezone.utc)
    today_str = today_dt.strftime("%Y-%m-%d")
    current_year = today_dt.year

    upcoming = []
    historical = []

    for league_id, comp_name in COMPETITIONS.items():
        # 1. Fetch meciuri de azi
        url_today = f"https://v3.football.api-sports.io/fixtures?date={today_str}&league={league_id}&season={current_year}"
        try:
            res = requests.get(url_today, headers=headers, timeout=5)
            if res.status_code == 200:
                for item in res.json().get("response", []):
                    upcoming.append({
                        "id": item["fixture"]["id"],
                        "league": comp_name,
                        "home": item["teams"]["home"]["name"],
                        "away": item["teams"]["away"]["name"],
                        "date": item["fixture"]["date"][:10],
                        "status": item["fixture"]["status"]["short"]
                    })
        except Exception:
            pass

        # 2. Fetch ultimele meciuri terminate (pentru învățare)
        url_hist = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={current_year}&last=30"
        try:
            res_h = requests.get(url_hist, headers=headers, timeout=5)
            if res_h.status_code == 200:
                for item in res_h.json().get("response", []):
                    if item["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]:
                        historical.append({
                            "home": item["teams"]["home"]["name"],
                            "away": item["teams"]["away"]["name"],
                            "score_home": item["goals"]["home"],
                            "score_away": item["goals"]["away"],
                            "date": item["fixture"]["date"][:10]
                        })
        except Exception:
            pass

    return historical, upcoming

# ============================================================
# 4. INTERFAȚĂ APLICAȚIE
# ============================================================

st.title("⚡ Quantum Analytics Engine — Year 3000 Edition")
st.caption("🤖 Predictor Autonom & Generator Biletul Zilei cu Auto-Învățare")

input_key = st.sidebar.text_input("🔑 Cheie API-Sports / RapidAPI:", value=st.session_state["api_key"], type="password")
if input_key:
    st.session_state["api_key"] = input_key

if not st.session_state["api_key"]:
    st.info("👈 Introdu cheia ta RapidAPI / API-Sports în bara laterală.")
else:
    with st.spinner("🔄 Se descarcă datele din API-Football v3..."):
        historical_matches, upcoming_matches = fetch_data_api_sports(st.session_state["api_key"])

    db, avg_home, avg_away = build_real_team_database(historical_matches)
    st.sidebar.success(f"🧠 {len(historical_matches)} meciuri analizate din sistem!")

    if not upcoming_matches:
        st.warning("⚠️ Nu sunt meciuri azi în ligile selectate sau cheia API este invalidă / fără credit.")
    else:
        match_map = {f"[{m['league']}] {m['home']} vs {m['away']}": m for m in upcoming_matches}
        selected_labels = st.sidebar.multiselect("Selectează meciuri de analizat:", options=list(match_map.keys()), default=list(match_map.keys()))

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
                "Pronostic": best_pick[0],
                "Încredere": f"{best_pick[1]*100:.1f}%",
                "Cotă Fair": f"{fair_odds(best_pick[1]):.2f}",
                "prob": best_pick[1], "odds": fair_odds(best_pick[1])
            })

        # Biletul Zilei Automat
        st.markdown("---")
        st.header("🎟️ BILETUL ZILEI AUTOMAT (Șansă Maximă de Reușită)")

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
                st.metric("Nivel Încredere", f"{avg_conf:.1f}%")

        # Analiza meciurilor
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
