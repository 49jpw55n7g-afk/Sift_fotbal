import csv
import math
import os
from datetime import datetime
import numpy as np
import requests
from scipy.stats import poisson
from understatapi import UnderstatClient

# ==========================================
# CONFIGURARE MEDIU & NIVELE DE SIGURANȚĂ
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # Cheia ta de la The-Odds-API (Opțional)

KELLY_FRACTION = 0.25  # Fractional Kelly 25% (Protecție împotriva volatilității)
MIN_VALUE_MARGIN = 0.03  # Minim +3% EV pentru a fi considerat Value Bet


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
    """Calcul miza optimă folosind Criteriul Fractional Kelly (% din bancă)."""
    b = bookmaker_odds - 1.0
    q = 1.0 - probability
    f_star = (b * probability - q) / b
    
    if f_star <= 0:
        return 0.0
    
    recommended_percentage = f_star * kelly_fraction * 100
    return round(min(recommended_percentage, 5.0), 2)  # Cap de siguranță la max 5% din bancă


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
# 2. PRELUARE DATE XG (EXPONENTIAL DECAY & HOME/AWAY SPLIT)
# ==========================================
def fetch_weighted_team_xg(season: int, team_name: str, is_home: bool, last_n: int = 5):
    """Preluare xG din Understat cu ponderare exponențială și filtrare pe meciuri acasă/deplasare."""
    understat = UnderstatClient()
    try:
        results = understat.get_team_results(team_name=team_name, season=season)
        played = [m for m in results if m.get('xG') is not None]
        
        # Filtrare strictă pe locație (Acasă sau Deplasare)
        location_matches = [
            m for m in played 
            if (m['h']['title'] == team_name if is_home else m['a']['title'] == team_name)
        ][-last_n:]

        if not location_matches:
            # Fallback pe ultimele meciuri generale dacă nu există suficiente meciuri pe locație
            location_matches = played[-last_n:]

        weights = [math.exp(i * 0.3) for i in range(len(location_matches))]  # Ponderare exponențială
        total_weight = sum(weights)

        scored_weighted = 0.0
        conceded_weighted = 0.0

        for idx, match in enumerate(location_matches):
            w = weights[idx]
            match_is_home = match['h']['title'] == team_name
            if match_is_home:
                scored_weighted += float(match['xG']['h']) * w
                conceded_weighted += float(match['xG']['a']) * w
            else:
                scored_weighted += float(match['xG']['a']) * w
                conceded_weighted += float(match['xG']['h']) * w

        return {
            "avg_xg_attack": round(scored_weighted / total_weight, 2),
            "avg_xg_conceded": round(conceded_weighted / total_weight, 2)
        }
    except Exception as e:
        print(f"Eroare preluare xG pentru {team_name}: {e}")
        return None


# ==========================================
# 3. INTEGRARE THE-ODDS-API (COTE LIVE)
# ==========================================
def fetch_live_odds(sport_key: str = "soccer_epl", region: str = "eu") -> dict:
    """Preia cotele reale directe de la casele de pariuri dacă există API Key."""
    if not ODDS_API_KEY:
        return {}
    
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={ODDS_API_KEY}&regions={region}&markets=h2h,totals"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Eroare preluare cote The-Odds-API: {e}")
        return {}


# ==========================================
# 4. TRIMITERE TELEGRAM & STOCARE CSV
# ==========================================
def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram netrimis: Lipsă Token sau Chat ID.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Eroare trimitere Telegram: {e}")


def save_value_bets_to_csv(home_team: str, away_team: str, analysis: dict, filepath: str = "history_value_bets.csv"):
    value_bets = analysis.get("Value Bets Detectate (+EV)", [])
    if not value_bets:
        return

    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "ID Meci", "Data Scanarii", "Meci", "Piata", 
                "Cota Casa", "Cota Reala", "EV Procent", 
                "Miza Kelly", "Status Pariu", "Profit/Loss"
            ])
            
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        match_name = f"{home_team} vs {away_team}"
        
        for vb in value_bets:
            match_id = f"{datetime.now().strftime('%Y%m%d')}_{home_team[:3]}_{away_team[:3]}_{vb['Piata'].replace(' ', '')}"
            writer.writerow([
                match_id,
                now_str,
                match_name,
                vb["Piata"],
                vb["Cota Casa"],
                vb["Cota Reala"],
                vb["EV (Margine Profit)"],
                vb["Miza Recomandata Kelly"],
                "PENDING",  # Va fi actualizat de scriptul de auto-resolver
                0.0
            ])


# ==========================================
# 5. EXECUȚIE SCANER
# ==========================================
def run_full_scanner(season: int, home_team: str, away_team: str, bookmaker_odds: dict):
    home_stats = fetch_weighted_team_xg(season, home_team, is_home=True)
    away_stats = fetch_weighted_team_xg(season, away_team, is_home=False)

    if not home_stats or not away_stats:
        return

    analysis = analyze_match_pro(
        home_xg_attack=home_stats['avg_xg_attack'],
        away_xg_attack=away_stats['avg_xg_attack'],
        home_xg_conceded=home_stats['avg_xg_conceded'],
        away_xg_conceded=away_stats['avg_xg_conceded'],
        bookmaker_odds=bookmaker_odds
    )

    value_bets = analysis.get("Value Bets Detectate (+EV)", [])
    if value_bets:
        save_value_bets_to_csv(home_team, away_team, analysis)
        
        # Construire mesaj Telegram
        msg = f"🚨 *VALUE BET DETECTAT (+EV)* 🚨\n\n"
        msg += f"⚽ *Meci:* {home_team} vs {away_team}\n"
        msg += f"📊 *xG Modelat:* {analysis['xG Modelat (Gazde - Oaspeti)']}\n\n"
        for vb in value_bets:
            msg += f"• *Pariu:* {vb['Piata']}\n"
            msg += f"  - Cotă Casă: `{vb['Cota Casa']}` | Cotă Reală: `{vb['Cota Reala']}`\n"
            msg += f"  - Profit Estimat: *{vb['EV (Margine Profit)']}*\n"
            msg += f"  - Miză Recomandată: *{vb['Miza Recomandata Kelly']}*\n\n"
            
        send_telegram_alert(msg)


if __name__ == "__main__":
    # Testare scanare
    odds_test = {
        "1 Solist": 2.25,
        "Peste 2.5 Goluri": 1.95,
        "Ambele Marcheaza (GG)": 1.85
    }
    run_full_scanner(season=2025, home_team="Arsenal", away_team="Chelsea", bookmaker_odds=odds_test)
