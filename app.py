import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import poisson
from datetime import datetime, timezone

# ============================================================
# 1. CONFIGURARE PAGINĂ STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Quantitative Football Analytics Engine V4",
    page_icon="⚽",
    layout="wide"
)

MAX_GOALS = 15
DECAY_DAYS = 180
SHRINKAGE_GAMES = 8
DEFAULT_RHO = -0.10

MIN_EDGE = 0.04
MIN_EV = 0.05

# Lista codurilor de competiție disponibile în planul gratuit football-data.org
FREE_COMPETITIONS = ["CL", "PL", "PD", "SA", "BL1", "FL1", "PPD", "DED"]

# ============================================================
# 2. UTILS & MATEMATICĂ (LOGICA TA INTACTĂ)
# ============================================================

def clamp(x, low, high):
    return max(low, min(high, float(x)))

def safe_probability(p):
    return clamp(p, 1e-6, 1.0 - 1e-6)

def fair_odds(prob):
    return 1.0 / safe_probability(prob)

def implied_probability(odds):
    if odds is None or odds <= 1.0:
        return None
    return 1.0 / odds

def calculate_ev(prob, odds):
    if odds is None or odds <= 1.0:
        return None
    return (prob * (odds - 1.0)) - (1.0 - prob)

# ============================================================
# 3. ENGINE DIXON-COLES V4 (LOGICA TA INTACTĂ)
# ============================================================

def get_temporal_weight(match_date, ref_date=None):
    if ref_date is None:
        ref_date = datetime.now(timezone.utc).replace(tzinfo=None)
    elif isinstance(ref_date, str):
        ref_date = datetime.fromisoformat(ref_date.replace("Z", "+00:00")).replace(tzinfo=None)

    if isinstance(match_date, str):
        try:
            match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return 0.5
    elif isinstance(match_date, datetime):
        match_date = match_date.replace(tzinfo=None)

    age_days = max(0, (ref_date - match_date).days)
    weight = math.exp(-age_days / DECAY_DAYS)
    return clamp(weight, 0.05, 1.0)

def build_real_team_database(matches, ref_date=None):
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
        date = m.get("date")
        weight = get_temporal_weight(date, ref_date)

        home_goals.append(sh)
        away_goals.append(sa)

        for team, venue, gf, ga in [(home, "home", sh, sa), (away, "away", sa, sh)]:
            if team not in teams:
                teams[team] = []
            teams[team].append({
                "venue": venue,
                "goals_for": gf,
                "goals_against": ga,
                "weight": weight
            })

    avg_home = float(np.mean(home_goals)) if home_goals else 1.40
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

    return {
        "attack": float(adj_gf / league_avg),
        "defense": float(adj_ga / league_avg),
        "games": games
    }

def calculate_recent_form(team, database, last_n=5):
    matches = database.get(team, [])
    if not matches:
        return 1.0

    sorted_m = sorted(matches, key=lambda x: x["weight"], reverse=True)[:last_n]
    points = []
    for m in sorted_m:
        if m["goals_for"] > m["goals_against"]:
            points.append(3)
        elif m["goals_for"] == m["goals_against"]:
            points.append(1)
        else:
            points.append(0)

    avg_p = np.mean(points) if points else 1.36
    return float(0.80 + (avg_p / 3.0) * 0.40)

def dixon_coles_tau(h, a, l_home, l_away, rho):
    if h == 0 and a == 0:
        return 1.0 - (l_home * l_away * rho)
    elif h == 0 and a == 1:
        return 1.0 + (l_home * rho)
    elif h == 1 and a == 0:
        return 1.0 + (l_away * rho)
    elif h == 1 and a == 1:
        return 1.0 - rho
    return 1.0

def build_dixon_coles_matrix(l_home, l_away, rho=DEFAULT_RHO, max_goals=MAX_GOALS):
    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base_p = poisson.pmf(h, l_home) * poisson.pmf(a, l_away)
            tau = dixon_coles_tau(h, a, l_home, l_away, rho)
            matrix[h, a] = base_p * tau

    total = np.sum(matrix)
    if total > 0:
        matrix /= total
    return matrix

def calculate_expected_goals(home_team, away_team, database, avg_home, avg_away):
    h_str = calculate_team_strength(home_team, "home", database, avg_home)
    a_str = calculate_team_strength(away_team, "away", database, avg_away)

    l_home = avg_home * h_str["attack"] * a_str["defense"]
    l_away = avg_away * a_str["attack"] * h_str["defense"]

    h_form = calculate_recent_form(home_team, database)
    a_form = calculate_recent_form(away_team, database)

    l_home *= (0.85 + 0.15 * h_form)
    l_away *= (0.85 + 0.15 * a_form)

    return clamp(l_home, 0.2, 4.5), clamp(l_away, 0.2, 4.5)

def extract_markets_from_matrix(matrix):
    size = matrix.shape[0]
    p_1 = float(np.sum(np.tril(matrix, -1)))
    p_x = float(np.sum(np.diag(matrix)))
    p_2 = float(np.sum(np.triu(matrix, 1)))

    p_btts_yes = float(np.sum(matrix[1:, 1:]))

    markets = {
        "1": p_1, "X": p_x, "2": p_2,
        "1X": p_1 + p_x, "X2": p_x + p_2, "12": p_1 + p_2,
        "BTTS_YES": p_btts_yes, "BTTS_NO": 1.0 - p_btts_yes
    }

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over_p = 0.0
        for h in range(size):
            for a in range(size):
                if h + a > line:
                    over_p += matrix[h, a]
        markets[f"OVER_{line}"] = float(over_p)
        markets[f"UNDER_{line}"] = float(1.0 - over_p)

    return markets

# ============================================================
# 4. PRELUARE AUTOMATĂ DIN API FOOTBALL-DATA.ORG
# ============================================================

@st.cache_data(ttl=43200)  # Reîmprospătare la 12 ore (în fiecare dimineață)
def fetch_api_matches(api_key):
    if not api_key or len(api_key.strip()) < 10:
        return [], []

    headers = {'X-Auth-Token': api_key.strip()}
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    upcoming = []
    historical = []

    # Interogare endpoint principal pentru meciurile de azi
    url = f"https://api.football-data.org/v4/matches?dateFrom={today_str}&dateTo={today_str}"
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        
        # Daca endpoint-ul global e blocat, parcurgem ligile gratuite permise
        if res.status_code in (400, 403):
            raw_matches = []
            for comp in FREE_COMPETITIONS:
                c_url = f"https://api.football-data.org/v4/competitions/{comp}/matches?dateFrom={today_str}&dateTo={today_str}"
                r = requests.get(c_url, headers=headers, timeout=5)
                if r.status_code == 200:
                    raw_matches.extend(r.json().get("matches", []))
        elif res.status_code == 200:
            raw_matches = res.json().get("matches", [])
        else:
            st.error(f"Eroare API: Cod HTTP {res.status_code}")
            return [], []

        for m in raw_matches:
            # Preluare cote furnizate de API dacă există, altfel cote implicite
            odds_data = m.get("odds", {})
            h_odds = odds_data.get("homeWin", 2.10)
            d_odds = odds_data.get("draw", 3.30)
            a_odds = odds_data.get("awayWin", 3.50)

            upcoming.append({
                "id": m.get("id"),
                "league": m.get("competition", {}).get("name", "Fotbal"),
                "home": m.get("homeTeam", {}).get("name", "Gazde"),
                "away": m.get("awayTeam", {}).get("name", "Oaspeți"),
                "date": m.get("utcDate", today_str)[:10],
                "odds": {"1": float(h_odds), "X": float(d_odds), "2": float(a_odds), "OVER_2.5": 1.90}
            })

            # Extragere meciuri finalizate (dacă există în răspuns) pentru baza de date
            if m.get("status") == "FINISHED":
                score = m.get("score", {}).get("fullTime", {})
                historical.append({
                    "home": m.get("homeTeam", {}).get("name"),
                    "away": m.get("awayTeam", {}).get("name"),
                    "score_home": score.get("home"),
                    "score_away": score.get("away"),
                    "completed": True,
                    "date": m.get("utcDate")[:10]
                })

        return historical, upcoming

    except Exception as e:
        st.error(f"Eroare la conectarea cu API-ul: {e}")
        return [], []

# ============================================================
# 5. INTERFAȚĂ STREAMLIT (UI)
# ============================================================

st.title("⚽ Quantitative Football Analytics Engine V4")
st.caption("Auto Value Bet Identification Engine | Conectat Live la API")

api_key = st.sidebar.text_input("Cheie API Football-Data.org", type="password")

if not api_key:
    st.info("👈 Introdu cheia API în bara laterală din stânga pentru a încărca automat meciurile de azi.")
else:
    historical_matches, upcoming_matches = fetch_api_matches(api_key)
    db, avg_home, avg_away = build_real_team_database(historical_matches)

    if not upcoming_matches:
        st.warning("Nu au fost găsite meciuri programate pentru astăzi în ligile din contul tău API.")
    else:
        st.sidebar.header("🎯 Filtru Meciuri")

        # Selecție pe Ligi
        leagues = sorted(list(set(m["league"] for m in upcoming_matches)))
        selected_leagues = st.sidebar.multiselect("Competiții:", options=leagues, default=leagues)

        filtered = [m for m in upcoming_matches if m["league"] in selected_leagues]

        # Selecție meciuri individuale
        match_map = {f"{m['league']} | {m['home']} vs {m['away']}": m['id'] for m in filtered}
        selected_labels = st.sidebar.multiselect("Alege meciurile de analizat:", options=list(match_map.keys()), default=list(match_map.keys()))

        selected_ids = [match_map[lbl] for lbl in selected_labels if lbl in match_map]
        matches_to_analyze = [m for m in filtered if m['id'] in selected_ids]

        st.subheader(f"📅 Meciuri Azi în Analiză ({len(matches_to_analyze)})")
        
        total_value_bets = 0

        for idx, match in enumerate(matches_to_analyze):
            home, away = match["home"], match["away"]
            league, match_date = match["league"], match["date"]

            l_h, l_a = calculate_expected_goals(home, away, db, avg_home, avg_away)
            matrix = build_dixon_coles_matrix(l_h, l_a)
            markets = extract_markets_from_matrix(matrix)

            with st.expander(f"🏆 {league} | {home} vs {away} ({match_date}) — xG: {l_h:.2f} - {l_a:.2f}"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    o1 = st.number_input("Cotă 1", value=match["odds"].get("1", 2.10), key=f"o1_{match['id']}")
                with c2:
                    ox = st.number_input("Cotă X", value=match["odds"].get("X", 3.30), key=f"ox_{match['id']}")
                with c3:
                    o2 = st.number_input("Cotă 2", value=match["odds"].get("2", 3.50), key=f"o2_{match['id']}")
                with c4:
                    o_over = st.number_input("Cotă Over 2.5", value=match["odds"].get("OVER_2.5", 1.90), key=f"oover_{match['id']}")

                current_odds = {"1": o1, "X": ox, "2": o2, "OVER_2.5": o_over}
                match_value_bets = []

                for m_key, odds in current_odds.items():
                    if m_key in markets:
                        prob = markets[m_key]
                        imp_prob = implied_probability(odds)
                        ev = calculate_ev(prob, odds)
                        edge = prob - imp_prob if imp_prob else 0

                        if (edge >= MIN_EDGE) and (ev >= MIN_EV):
                            match_value_bets.append({
                                "Piață": m_key,
                                "Prob. Model": f"{prob * 100:.1f}%",
                                "Cotă Fair": f"{fair_odds(prob):.2f}",
                                "Cotă Bookie": f"{odds:.2f}",
                                "Edge": f"{edge * 100:+.1f}%",
                                "EV": f"{ev * 100:+.1f}%"
                            })

                if match_value_bets:
                    total_value_bets += len(match_value_bets)
                    st.success("🟢 VALUE BET IDENTIFICAT!")
                    st.dataframe(pd.DataFrame(match_value_bets), use_container_width=True)
                else:
                    st.info("🔴 PASS — Fără valoare matematică (+EV) identificată.")

        st.sidebar.markdown("---")
        st.sidebar.metric("Total Value Bets (+EV)", total_value_bets)
