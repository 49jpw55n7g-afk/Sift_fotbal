import math
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import poisson
from datetime import datetime, timezone

# ============================================================
# CONFIGURARE PAGINĂ STREAMLIT
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
# MOCK HISTORICAL DATA CA EXEMPLU REAL DE START
# ============================================================

def get_sample_matches():
    return [
        {"home": "Inter", "away": "Milan", "score_home": 2, "score_away": 1, "completed": True, "date": "2026-08-01"},
        {"home": "Juventus", "away": "Roma", "score_home": 1, "score_away": 1, "completed": True, "date": "2026-08-02"},
        {"home": "Inter", "away": "Juventus", "score_home": 1, "score_away": 0, "completed": True, "date": "2026-08-10"},
        {"home": "Milan", "away": "Roma", "score_home": 3, "score_away": 2, "completed": True, "date": "2026-08-12"},
        {"home": "Roma", "away": "Inter", "score_home": 0, "score_away": 2, "completed": True, "date": "2026-08-20"},
        {"home": "Milan", "away": "Juventus", "score_home": 2, "score_away": 2, "completed": True, "date": "2026-08-22"},
        {"home": "Inter", "away": "Napoli", "score_home": 3, "score_away": 1, "completed": True, "date": "2026-08-28"},
        {"home": "Napoli", "away": "Milan", "score_home": 1, "score_away": 2, "completed": True, "date": "2026-09-01"},
    ]

# ============================================================
# INTERFAȚĂ STREAMLIT (UI)
# ============================================================

st.title("⚽ Quantitative Football Analytics Engine V4")
st.caption("Dixon-Coles + Weighted MLE + Expected Value (EV) Filtering")

matches = get_sample_matches()
db, avg_home, avg_away = build_real_team_database(matches)
available_teams = sorted(list(db.keys()))

st.sidebar.header("⚙️ Configurare Meci")

if len(available_teams) >= 2:
    home_team = st.sidebar.selectbox("Echipa Gazdă", available_teams, index=0)
    away_team = st.sidebar.selectbox("Echipa Oaspete", available_teams, index=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Cote Bookmaker")
    odds_1 = st.sidebar.number_input("Cotă 1", value=1.95, step=0.05)
    odds_x = st.sidebar.number_input("Cotă X", value=3.40, step=0.05)
    odds_2 = st.sidebar.number_input("Cotă 2", value=4.10, step=0.05)
    odds_over25 = st.sidebar.number_input("Cotă Over 2.5", value=2.05, step=0.05)

    if home_team == away_team:
        st.error("Selectează două echipe diferite!")
    else:
        l_h, l_a = calculate_expected_goals(home_team, away_team, db, avg_home, avg_away)
        matrix = build_dixon_coles_matrix(l_h, l_a)
        markets = extract_markets_from_matrix(matrix)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expected Goals Gazde", f"{l_h:.2f}")
        with col2:
            st.metric("Expected Goals Oaspeți", f"{l_a:.2f}")

        st.subheader("📊 Analiză Valoare Matematică (Value Bets)")

        bookmaker_odds = {
            "1": odds_1,
            "X": odds_x,
            "2": odds_2,
            "OVER_2.5": odds_over25
        }

        results = []
        for market, prob in markets.items():
            if market in bookmaker_odds:
                odds = bookmaker_odds[market]
                imp_prob = implied_probability(odds)
                ev = calculate_ev(prob, odds)
                edge = prob - imp_prob if imp_prob else 0

                is_value = (edge >= MIN_EDGE) and (ev >= MIN_EV)
                decision = "🟢 VALUE BET" if is_value else "🔴 PASS"

                results.append({
                    "Piață": market,
                    "Probabilitate Model": f"{prob * 100:.1f}%",
                    "Cotă Corectă (Fair)": f"{fair_odds(prob):.2f}",
                    "Cotă Bookmaker": f"{odds:.2f}",
                    "Prob. Implicată": f"{imp_prob * 100:.1f}%" if imp_prob else "N/A",
                    "Edge": f"{edge * 100:+.1f}%",
                    "EV": f"{ev * 100:+.1f}%",
                    "Decizie": decision
                })

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

else:
    st.info("Sunt necesare cel puțin două echipe în baza de date pentru evaluare.")
