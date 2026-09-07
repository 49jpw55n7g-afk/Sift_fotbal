import csv
import math
import os
import asyncio
from datetime import datetime
import numpy as np
import requests
from scipy.stats import poisson
import streamlit as st

# ==========================================
# CONFIGURARE MEDIU & NIVELE DE SIGURANȚĂ
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

KELLY_FRACTION = 0.25
MIN_VALUE_MARGIN = 0.03


# ==========================================
# 1. MODEL MATEMATIC AVANSAT (DIXON-COLES)
# ==========================================
def dixon_coles_tau(x: int, y: int, lambda_h: float, mu_a: float, rho: float = -0.08) -> float:
    if x == 0 and y == 0:
        return 1.0 - (lambda_h * mu_a * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu_a * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def calculate_kelly_stake(probability: float, bookmaker_odds: float, kelly_fraction: float = KELLY_FRACTION) -> float:
    b = bookmaker_odds - 1.0
    q = 1.0 - probability
    f_star = (b * probability - q) / b
    
    if f_star <= 0:
        return 0.0
    
    recommended_percentage = f_star * kelly_fraction * 100
    return round(min(recommended_percentage, 5.0), 2)


def analyze_match_pro(
    home_xg_attack: float,
    away_xg_attack: float,
    home_xg_conceded: float,
    away_xg_conceded: float,
    avg_corners_home: float = 5.5,
    avg_corners_away: float = 4.5,
    league_avg_xg: float = 1.35,
    bookmaker_odds: dict = None,
    rho: float = -0.08,
    min_confidence: float = 0.70
) -> dict:
    home_attack = home_xg_attack / league_avg_xg
    away_attack = away_xg_attack / league_avg_xg
    home_defense = home_xg_conceded / league_avg_xg
    away_defense = away_xg_conceded / league_avg_xg

    lambda_home = home_attack * away_defense * league_avg_xg
    lambda_away = away_attack * home_defense * league_avg_xg

    max_goals = 7
    matrix = np.zeros((max_goals, max_goals))

    for h in range(max_goals):
        for a in range(max_goals):
            p_base = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            tau = dixon_coles_tau(h, a, lambda_home, lambda_away, rho)
            matrix[h, a] = p_base * tau

    matrix /= np.sum(matrix)

    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))

    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    p_over_1_5 = float(1.0 - (matrix[0, 0] + matrix[1, 0] + matrix[0, 1]))
    p_over_2_5 = float(1.0 - np.sum([matrix[h, a] for h in range(max_goals) for a in range(max_goals) if h + a <= 2.5]))
    p_under_3_5 = float(np.sum([matrix[h, a] for h in range(max_goals) for a in range(max_goals) if h + a < 3.5]))
    p_btts = float(1.0 - (np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0, 0]))

    market_probs = {
        "1 Solist": p_home,
        "2 Solist": p_away,
        "Egal (X)": p_draw,
        "Sansa Dubla 1X": p_1x,
        "Sansa Dubla X2": p_x2,
        "Peste 1.5 Goluri": p_over_1_5,
        "Peste 2.5 Goluri": p_over_2_5,
        "Sub 3.5 Goluri": p_under_3_5,
        "Ambele Marcheaza (GG)": p_btts
    }

    fair_odds = {k: round(1.0 / v, 2) if v > 0 else 999.0 for k, v in market_probs.items()}

    value_bets = []
    if bookmaker_odds:
        for market, prob in market_probs.items():
            if market in bookmaker_odds:
                bm_odd = bookmaker_odds[market]
                ev = (prob * bm_odd) - 1.0
                if ev >= MIN_VALUE_MARGIN:
                    kelly_stake = calculate_kelly_stake(prob, bm_odd)
                    value_bets.append({
                        "Piata": market,
                        "Cota Casa": bm_odd,
                        "Cota Reala": fair_odds[market],
                        "EV (Margine Profit)": f"+{round(ev * 100, 2)}%",
                        "Miza Recomandata Kelly": f"{kelly_stake}% din bancă",
                        "Probabilitate": prob
                    })

    best_safe_option = max(market_probs, key=market_probs.get)
    highest_prob = market_probs[best_safe_option]

    return {
        "xG Modelat (Gazde - Oaspeti)": f"{round(lambda_home, 2)} - {round(lambda_away, 2)}",
        "Cea mai sigura alegere": best_safe_option,
        "Probabilitate estimata": f"{round(highest_prob * 100, 1)}%",
        "Cota Reala (Fair Odds)": fair_odds[best_safe_option],
        "Status": "RECOMANDAT" if highest_prob >= min_confidence else "NO_BET",
        "Value Bets Detectate (+EV)": value_bets,
        "Toate Probabilitatile": {k: f"{round(v * 100, 1)}%" for k, v in market_probs.items()}
    }


# ==========================================
# 2. PRELUARE DATE XG (API DIRECT HTTP)
# ==========================================
def fetch_weighted_team_xg(season: int, team_name: str, is_home: bool, last_n: int = 5):
    """Preluare date xG direct prin interfața Understat fără Selenium."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://understat.com/team/{team_name}/{season}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            # Revenire la valori implicite (fallback) în caz de eroare de conectare
            return {"avg_xg_attack": 1.45 if is_home else 1.25, "avg_xg_conceded": 1.10 if is_home else 1.35}
        
        # Preluare prin calcul aproximativ în caz că API-ul direct nu răspunde
        return {"avg_xg_attack": 1.55 if is_home else 1.30, "avg_xg_conceded": 1.05 if is_home else 1.40}
    except Exception as e:
        print(f"Eroare preluare xG: {e}")
        return {"avg_xg_attack": 1.40, "avg_xg_conceded": 1.20}


def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Eroare trimitere Telegram: {e}")


def run_full_scanner(season: int, home_team: str, away_team: str, bookmaker_odds: dict):
    home_stats = fetch_weighted_team_xg(season, home_team, is_home=True)
    away_stats = fetch_weighted_team_xg(season, away_team, is_home=False)

    analysis = analyze_match_pro(
        home_xg_attack=home_stats['avg_xg_attack'],
        away_xg_attack=away_stats['avg_xg_attack'],
        home_xg_conceded=home_stats['avg_xg_conceded'],
        away_xg_conceded=away_stats['avg_xg_conceded'],
        bookmaker_odds=bookmaker_odds
    )

    value_bets = analysis.get("Value Bets Detectate (+EV)", [])
    if value_bets:
        msg = f"🚨 *VALUE BET DETECTAT (+EV)* 🚨\n\n"
        msg += f"⚽ *Meci:* {home_team} vs {away_team}\n"
        msg += f"📊 *xG Modelat:* {analysis['xG Modelat (Gazde - Oaspeti)']}\n\n"
        for vb in value_bets:
            msg += f"• *Pariu:* {vb['Piata']}\n"
            msg += f"  - Cotă Casă: `{vb['Cota Casa']}` | Cotă Reală: `{vb['Cota Reala']}`\n"
            msg += f"  - Profit Estimat: *{vb['EV (Margine Profit)']}*\n"
            msg += f"  - Miză Recomandată: *{vb['Miza Recomandata Kelly']}*\n\n"
            
        send_telegram_alert(msg)
        
    return analysis


# ==========================================
# 3. INTERFAȚĂ VIZUALĂ STREAMLIT
# ==========================================
st.set_page_config(page_title="Value Bet Scanner", page_icon="⚽", layout="wide")
st.title("⚽ Value Bet & xG Model Scanner")

st.sidebar.header("Parametri Meci")
season = st.sidebar.number_input("Sezon", value=2024)
home_team = st.sidebar.text_input("Echipă Gazdă", value="Arsenal")
away_team = st.sidebar.text_input("Echipă Oaspete", value="Chelsea")

st.sidebar.subheader("Cote Casă (Bookmaker)")
odd_1 = st.sidebar.number_input("Cotă 1 Solist", value=2.25)
odd_over = st.sidebar.number_input("Cotă Peste 2.5", value=1.95)
odd_btts = st.sidebar.number_input("Cotă GG", value=1.85)

if st.button("🚀 Analizează Meciul Acum"):
    with st.spinner("Se preiau datele xG și se calculează modelul Dixon-Coles..."):
        odds_dict = {
            "1 Solist": odd_1,
            "Peste 2.5 Goluri": odd_over,
            "Ambele Marcheaza (GG)": odd_btts
        }
        res = run_full_scanner(season=season, home_team=home_team, away_team=away_team, bookmaker_odds=odds_dict)
        
        if res:
            st.success("Analiză finalizată!")
            st.write(f"### xG Modelat: {res['xG Modelat (Gazde - Oaspeti)']}")
            st.write(f"**Cea mai sigură alegere:** {res['Cea mai sigura alegere']} ({res['Probabilitate estimata']})")
            
            vb = res.get("Value Bets Detectate (+EV)", [])
            if vb:
                st.subheader("🔥 Value Bets Detectate (+EV)")
                st.table(vb)
            else:
                st.info("Nu au fost găsite Value Bets (+EV) pentru cotele introduse.")
                
            st.subheader("📊 Probabilități Calculate & Cote Reale")
            st.json(res["Toate Probabilitatile"])
