import json
import re
import math
import requests
import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Predicții xG Meciuri", page_icon="⚽", layout="wide")

st.title("⚽ Predicții Automate xG - Top 5 Ligi")
st.markdown("Aplicația preia automat datele actualizate xG și îți permite să simulezi meciurile din top 5 ligi europene.")

# --- MAPPING LIGI UNDERSTAT ---
LEAGUES = {
    "Premier League (Anglia)": "EPL",
    "La Liga (Spania)": "La_Liga",
    "Bundesliga (Germania)": "Bundesliga",
    "Serie A (Italia)": "Serie_A",
    "Ligue 1 (Franța)": "Ligue_1"
}

# --- FUNCTII DE SCRAPING SI CALCUL ---
@st.cache_data(ttl=3600)
def get_league_data(league_code):
    """Extrage datele despre echipe și meciuri din Understat"""
    url = f"https://understat.com/league/{league_code}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Extrage datele echipelor
        teams_match = re.search(r"teamsData\s*=\s*JSON\.parse\('([^']+)'\)", response.text)
        dates_match = re.search(r"datesData\s*=\s*JSON\.parse\('([^']+)'\)", response.text)
        
        if not teams_match or not dates_match:
            return None, None
            
        teams_json = bytes(teams_match.group(1), 'utf-8').decode('unicode_escape')
        dates_json = bytes(dates_match.group(1), 'utf-8').decode('unicode_escape')
        
        teams_data = json.loads(teams_json)
        dates_data = json.loads(dates_json)
        
        return teams_data, dates_data
    except Exception as e:
        st.error(f"Eroare la preluarea datelor pentru {league_code}: {e}")
        return None, None

def calculate_team_stats(teams_data):
    """Calculează media de atac/defensivă xG pentru fiecare echipă și media ligii"""
    stats = {}
    total_xg_scored = 0
    total_xg_conceded = 0
    total_games = 0

    for team_id, team in teams_data.items():
        team_name = team['title']
        history = team['history']
        
        played_games = len(history)
        if played_games == 0:
            continue
            
        xg_scored = sum(float(m['xG']) for m in history)
        xg_conceded = sum(float(m['xGA']) for m in history)
        
        avg_xg_scored = xg_scored / played_games
        avg_xg_conceded = xg_conceded / played_games
        
        stats[team_name] = {
            "avg_xg_scored": avg_xg_scored,
            "avg_xg_conceded": avg_xg_conceded,
            "games": played_games
        }
        
        total_xg_scored += xg_scored
        total_xg_conceded += xg_conceded
        total_games += played_games

    league_avg_xg = (total_xg_scored / total_games) if total_games > 0 else 1.35
    return stats, league_avg_xg

def calculate_match_probabilities(home_team, away_team, stats, league_avg):
    """Calculează golurile așteptate și matricea de probabilități Poisson"""
    home_attack = stats[home_team]["avg_xg_scored"] / league_avg
    home_defense = stats[home_team]["avg_xg_conceded"] / league_avg
    
    away_attack = stats[away_team]["avg_xg_scored"] / league_avg
    away_defense = stats[away_team]["avg_xg_conceded"] / league_avg
    
    # Avantaj teren propriu (aprox +12%)
    exp_home_goals = home_attack * away_defense * league_avg * 1.12
    exp_away_goals = away_attack * home_defense * league_avg
    
    # Matrice de scoruri (0-8 goluri)
    max_goals = 8
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = poisson.pmf(i, exp_home_goals) * poisson.pmf(j, exp_away_goals)
            
    prob_home = np.sum(np.tril(matrix, -1))
    prob_draw = np.sum(np.diag(matrix))
    prob_away = np.sum(np.triu(matrix, 1))
    
    return exp_home_goals, exp_away_goals, prob_home, prob_draw, prob_away, matrix

# --- INTERFAȚA UTILIZATOR ---

col_l, col_m = st.columns([1, 2])

with col_l:
    selected_league_label = st.selectbox("Alege Liga:", list(LEAGUES.keys()))
    league_code = LEAGUES[selected_league_label]

    teams_data, dates_data = get_league_data(league_code)

    if teams_data and dates_data:
        stats, league_avg = calculate_team_stats(teams_data)
        
        # Filtrează meciurile viitoare / nejucate
        upcoming_matches = [m for m in dates_data if not m.get('isResult', True)]
        
        st.subheader("📌 Meciuri Disponibile")
        
        # Opțiune de selecție meci automat sau manual
        mode = st.radio("Sursă meci:", ["Din meciurile viitoare ale ligii", "Selecție manuală echipă vs echipă"])
        
        if mode == "Din meciurile viitoare ale ligii" and upcoming_matches:
            match_options = [f"{m['h']['title']} vs {m['a']['title']} ({m['datetime'][:10]})" for m in upcoming_matches]
            selected_match_str = st.selectbox("Alege meciul:", match_options)
            
            idx = match_options.index(selected_match_str)
            match_info = upcoming_matches[idx]
            home_team = match_info['h']['title']
            away_team = match_info['a']['title']
        else:
            team_list = sorted(list(stats.keys()))
            home_team = st.selectbox("Echipa Gazdă:", team_list, index=0)
            away_team = st.selectbox("Echipa Oaspete:", team_list, index=min(1, len(team_list)-1))

with col_m:
    if teams_data and stats and home_team != away_team:
        exp_home, exp_away, p_home, p_draw, p_away, matrix = calculate_match_probabilities(
            home_team, away_team, stats, league_avg
        )
        
        st.header(f"{home_team} vs {away_team}")
        
        # Metrică Goluri Așteptate
        c1, c2 = st.columns(2)
        c1.metric(f"Goluri Așteptate {home_team} (xG)", f"{exp_home:.2f}")
        c2.metric(f"Goluri Așteptate {away_team} (xG)", f"{exp_away:.2f}")
        
        st.subheader("📊 Probabilități Rezultat Final (1X2)")
        col_1, col_x, col_2 = st.columns(3)
        col_1.metric(f"Victorie {home_team} (1)", f"{p_home*100:.1f}%", f"Cota: {1/p_home:.2f}" if p_home > 0 else "")
        col_x.metric("Egalitate (X)", f"{p_draw*100:.1f}%", f"Cota: {1/p_draw:.2f}" if p_draw > 0 else "")
        col_2.metric(f"Victorie {away_team} (2)", f"{p_away*100:.1f}%", f"Cota: {1/p_away:.2f}" if p_away > 0 else "")
        
        # Probabilități Peste/Sub 2.5 Goluri
        st.subheader("⚽ Goluri Totale (Peste/Sub)")
        over_25_prob = 1 - sum(matrix[i, j] for i in range(3) for j in range(3) if i + j <= 2)
        under_25_prob = 1 - over_25_prob
        
        g1, g2 = st.columns(2)
        g1.metric("Peste 2.5 Goluri", f"{over_25_prob*100:.1f}%", f"Cota: {1/over_25_prob:.2f}" if over_25_prob > 0 else "")
        g2.metric("Sub 2.5 Goluri", f"{under_25_prob*100:.1f}%", f"Cota: {1/under_25_prob:.2f}" if under_25_prob > 0 else "")
        
        # Cele mai probabile scoruri corecte
        st.subheader("🎯 Cele mai probabile Scoruri Corecte")
        scores = []
        for i in range(5):
            for j in range(5):
                scores.append(((i, j), matrix[i, j]))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        top_scores = scores[:5]
        cols = st.columns(5)
        for idx, (sc, prob) in enumerate(top_scores):
            cols[idx].metric(f"Scor {sc[0]}-{sc[1]}", f"{prob*100:.1f}%", f"Cota: {1/prob:.2f}" if prob > 0 else "")
