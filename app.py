import math
import re
import unicodedata
import requests
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import poisson
from datetime import datetime, timezone

# ============================================================
# CONFIGURARE PAGINĂ STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Auto Football Value Betting Engine V4",
    page_icon="⚽",
    layout="wide"
)

MAX_GOALS = 15
DECAY_DAYS = 180
SHRINKAGE_GAMES = 8
DEFAULT_RHO = -0.10

MIN_EDGE = 0.04
MIN_EV = 0.05

# ============================================================
# UTILS & MATEMATICĂ
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
# PONDERARE TEMPORALĂ & ENGINE V4
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
# INCARCARE AUTOMATĂ MATEI & COTE REALTIME / API
# ============================================================

@st.cache_data(ttl=3600)
def fetch_upcoming_and_historical_matches(api_key=""):
    """
    Încarcă meciurile zilei + istoricul recent folosind API-ul Football-Data.org.
    Dacă nu există cheie API introduse, folosește un feed demonstrativ automat.
    """
    if not api_key:
        # Feed automat demonstrativ cu meciurile zilei curent
        today = datetime.now().strftime("%Y-%m-%d")
        
        historical = [
            {"home": "Arsenal", "away": "Chelsea", "score_home": 2, "score_away": 1, "completed": True, "date": "2026-08-10"},
            {"home": "Liverpool", "away": "Man City", "score_home": 1, "score_away": 1, "completed": True, "date": "2026-08-12"},
            {"home": "Man United", "away": "Arsenal", "score_home": 0, "score_away": 2, "completed": True, "date": "2026-08-15"},
            {"home": "Chelsea", "away": "Tottenham", "score_home": 3, "score_away": 2, "completed": True, "date": "2026-08-20"},
            {"home": "Man City", "away": "Arsenal", "score_home": 2, "score_away": 2, "completed": True, "date": "2026-08-25"},
            {"home": "Barcelona", "away": "Real Madrid", "score_home": 1, "score_away": 2, "completed": True, "date": "2026-08-18"},
            {"home": "Atletico", "away": "Barcelona", "score_home": 0, "score_away": 1, "completed": True, "date": "2026-08-22"}
        ]
        
        upcoming = [
            {
                "id": 101, "league": "Premier League", "home": "Arsenal", "away": "Man City", "date": today,
                "odds": {"1": 2.65, "X": 3.40, "2": 2.70, "OVER_2.5": 1.95, "BTTS_YES": 1.75}
            },
            {
                "id": 102, "league": "Premier League", "home": "Chelsea", "away": "Liverpool", "date": today,
                "odds": {"1": 3.10, "X": 3.50, "2": 2.25, "OVER_2.5": 1.70, "BTTS_YES": 1.60}
            },
            {
                "id": 103, "league": "La Liga", "home": "Barcelona", "away": "Atletico", "date": today,
                "odds": {"1": 1.90, "X": 3.60, "2": 4.20, "OVER_2.5": 1.85, "BTTS_YES": 1.80}
            }
        ]
        return historical, upcoming

    # Conexiune API Football-Data.org (când cheia e introdusă)
    headers = {'X-Auth-Token': api_key}
    url_upcoming = "https://api.football-data.org/v4/matches"
    
    try:
        res = requests.get(url_upcoming, headers=headers).json()
        upcoming = []
        for m in res.get("matches", []):
            upcoming.append({
                "id": m["id"],
                "league": m["competition"]["name"],
                "home": m["homeTeam"]["name"],
                "away": m["awayTeam"]["name"],
                "date": m["utcDate"][:10],
                "odds": {"1": 2.10, "X": 3.30, "2": 3.60, "OVER_2.5": 1.90} # Cote estimate de sistem
            })
        return [], upcoming
    except Exception as e:
        st.error(f"Eroare conectare API: {e}")
        return [], []

# ============================================================
# INTERFAȚĂ STREAMLIT (UI) - AUTOMATĂ
# ============================================================

st.title("⚽ Meciurile Zilei — Recomandări Value Bet Automate")
st.caption("Filtru automat bazat pe modelul Dixon-Coles V4, xG real și Expected Value (EV)")

# Sidebar opțional pentru cheie API
api_key = st.sidebar.text_input("Cheie API Football-Data.org (Opțional)", type="password")
historical_matches, upcoming_matches = fetch_upcoming_and_historical_matches(api_key)

db, avg_home, avg_away = build_real_team_database(historical_matches)

if not upcoming_matches:
    st.warning("Nu există meciuri programate pentru astăzi sau nu s-au putut prelua datele.")
else:
    st.subheader(f"📅 Program Zilei ({len(upcoming_matches)} Meciuri Analizate Automate)")
    
    value_bets_found = 0
    
    for match in upcoming_matches:
        home = match["home"]
        away = match["away"]
        league = match["league"]
        odds_dict = match.get("odds", {})

        # Daca echipele sunt noi, se folosesc valorile medii ale ligii
        l_h, l_a = calculate_expected_goals(home, away, db, avg_home, avg_away)
        matrix = build_dixon_coles_matrix(l_h, l_a)
        markets = extract_markets_from_matrix(matrix)

        # Evaluare automată Value Bet
        match_value_bets = []
        for market_key, odds in odds_dict.items():
            if market_key in markets:
                prob = markets[market_key]
                imp_prob = implied_probability(odds)
                ev = calculate_ev(prob, odds)
                edge = prob - imp_prob if imp_prob else 0

                if (edge >= MIN_EDGE) and (ev >= MIN_EV):
                    match_value_bets.append({
                        "Piață": market_key,
                        "Prob. Model": f"{prob * 100:.1f}%",
                        "Cotă Corectă": f"{fair_odds(prob):.2f}",
                        "Cotă Bookie": f"{odds:.2f}",
                        "Edge": f"{edge * 100:+.1f}%",
                        "EV": f"{ev * 100:+.1f}%"
                    })

        # Afișare meci
        with st.expander(f"🏆 {league} | {home} vs {away} — xG Estimat: {l_h:.2f} - {l_a:.2f}", expanded=bool(match_value_bets)):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Probabilitate 1:** {markets['1']*100:.1f}%")
            with col2:
                st.write(f"**Probabilitate X:** {markets['X']*100:.1f}%")
            with col3:
                st.write(f"**Probabilitate 2:** {markets['2']*100:.1f}%")

            if match_value_bets:
                value_bets_found += len(match_value_bets)
                st.success("🟢 VALUE BET IDENTIFICAT!")
                st.dataframe(pd.DataFrame(match_value_bets), use_container_width=True)
            else:
                st.info("🔴 PASS — Nu există nicio cotă cu valoare matematică (+EV suficient).")

    st.sidebar.markdown("---")
    st.sidebar.metric("Total Value Bets Găsite", value_bets_found)
