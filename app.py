import math
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import minimize
from scipy.stats import poisson
from difflib import SequenceMatcher
from datetime import datetime, timezone

# ============================================================
# CONFIGURARE SENSITIVITATE ȘI PARAMETRI
# ============================================================

MAX_GOALS = 15
DECAY_DAYS = 180           # Ponderare temporală (jumătate de viață a datelor)
SHRINKAGE_GAMES = 8        # Ponderare spre media ligii pentru eșantioane mici
DEFAULT_RHO = -0.10        # Corecție Dixon-Coles implicită pentru scoruri joase (0-0, 1-0, 0-1, 1-1)

# Praguri minime pentru a clasifica un pariu ca fiind VALUE BET
MIN_EDGE = 0.04            # Edge de minim 4% față de cota implicată
MIN_EV = 0.05              # Expected Value minim de +5%

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
    """Calcul Expected Value (EV) raportat la 1 unitate mizată."""
    if odds is None or odds <= 1.0:
        return None
    return (prob * (odds - 1.0)) - (1.0 - prob)

# ============================================================
# PONDERARE TEMPORALĂ (TIME DECAY)
# ============================================================

def get_temporal_weight(match_date, ref_date=None):
    """
    Calculează greutatea unui meci în funcție de cât de recent a fost jucat.
    Previne Data Leakage primind opțional un ref_date (data simulării).
    """
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

# ============================================================
# CONSTRUIRE DATABASE REAL & LEAGUE AVERAGES (FĂRĂ MOCK DATA)
# ============================================================

def build_real_team_database(matches, ref_date=None):
    """
    Procesează meciurile reale. Elimină orice generație aleatorie.
    """
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

# ============================================================
# CALCUL STRENGTH & SHRINKAGE
# ============================================================

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

    # Bayesian Shrinkage către media ligii pentru stabilizare
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

    # Sortare după recență
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
    # Normalizare între 0.80 și 1.20
    return float(0.80 + (avg_p / 3.0) * 0.40)

# ============================================================
# DIXON-COLES CORE ENGINE
# ============================================================

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

    # Ajustare moderată cu forma
    h_form = calculate_recent_form(home_team, database)
    a_form = calculate_recent_form(away_team, database)

    l_home *= (0.85 + 0.15 * h_form)
    l_away *= (0.85 + 0.15 * a_form)

    return clamp(l_home, 0.2, 4.5), clamp(l_away, 0.2, 4.5)

# ============================================================
# PIEȚE PROBABILISTICE & VALUE EVALUATION
# ============================================================

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

def evaluate_value_bet(market_key, model_prob, bookmaker_odds):
    if bookmaker_odds is None or bookmaker_odds <= 1.0:
        return None

    implied_p = implied_probability(bookmaker_odds)
    ev = calculate_ev(model_prob, bookmaker_odds)
    edge = model_prob - implied_p

    is_value = (edge >= MIN_EDGE) and (ev >= MIN_EV)

    return {
        "market": market_key,
        "model_prob": model_prob,
        "fair_odds": fair_odds(model_prob),
        "bookie_odds": bookmaker_odds,
        "implied_prob": implied_p,
        "edge": edge,
        "ev": ev,
        "is_value": is_value
    }

# ============================================================
# STRICT BACKTESTING FRAMEWORK (PREVENIRE DATA LEAKAGE)
# ============================================================

def run_leakage_free_backtest(historical_matches):
    """
    Execută un backtest unde la fiecare meci se folosesc
    EXCLUSIV meciurile jucate ANTERIOR datei meciului analizat.
    """
    # Sortare cronologică
    sorted_matches = sorted(
        [m for m in historical_matches if m.get("completed")],
        key=lambda x: x["date"]
    )

    brier_scores = []
    log_losses = []
    evaluated_bets = []

    for i in range(20, len(sorted_matches)):  # Pornim după un minim de meciuri
        target = sorted_matches[i]
        past_matches = sorted_matches[:i]

        ref_date = target["date"]
        db, avg_h, avg_a = build_real_team_database(past_matches, ref_date=ref_date)

        h_team, a_team = target["home"], target["away"]
        if h_team not in db or a_team not in db:
            continue

        l_h, l_a = calculate_expected_goals(h_team, a_team, db, avg_h, avg_a)
        matrix = build_dixon_coles_matrix(l_h, l_a)
        markets = extract_markets_from_matrix(matrix)

        # Rezultat real (0: 1, 1: X, 2: 2)
        sh, sa = float(target["score_home"]), float(target["score_away"])
        actual_idx = 0 if sh > sa else (1 if sh == sa else 2)
        probs_1x2 = [markets["1"], markets["X"], markets["2"]]

        # Metrics
        brier = sum((probs_1x2[j] - (1 if j == actual_idx else 0)) ** 2 for j in range(3))
        log_loss = -math.log(safe_probability(probs_1x2[actual_idx]))

        brier_scores.append(brier)
        log_losses.append(log_loss)

    return {
        "mean_brier": float(np.mean(brier_scores)) if brier_scores else None,
        "mean_log_loss": float(np.mean(log_losses)) if log_losses else None,
        "evaluated_matches": len(brier_scores)
    }
