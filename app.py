import json
import re
from datetime import datetime, timedelta
import requests
import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Predicții xG Meciuri", page_icon="⚽", layout="wide")

st.title("⚽ Predicții Automate xG - Meciurile Zilei (Live & Programate)")

LEAGUES = {
    "🇮🇹 Serie A (Italia)": "Serie_A",
    "🇪🇸 La Liga (Spania)": "La_Liga",
    "🇷🇴 Superliga (România)": "ROU_1",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Anglia)": "EPL",
    "🇩🇪 Bundesliga (Germania)": "Bundesliga",
    "🇫🇷 Ligue 1 (Franța)": "Ligue_1"
}

# Program Runda Curentă / Azi Superliga România
ROMANIA_FIXTURES = [
    {"h": "Universitatea Craiova", "a": "U Cluj", "time": "Azi - 20:30", "status": "AZI"},
    {"h": "FC Voluntari", "a": "FC Argeș Pitești", "time": "Azi - Finalizat", "status": "JUCAT"},
    {"h": "FCSB", "a": "Dinamo București", "time": "Etapa Curentă - 20:30", "status": "VIITOR"},
    {"h": "CFR Cluj", "a": "FC Botoșani", "time": "Etapa Curentă - 18:00", "status": "VIITOR"},
    {"h": "Rapid București", "a": "Farul Constanța", "time": "Etapa Curentă - 20:00", "status": "VIITOR"}
]

ROMANIA_TEAMS = {
    "FCSB": {"avg_xg_scored": 1.65, "avg_xg_conceded": 1.10},
    "CFR Cluj": {"avg_xg_scored": 1.55, "avg_xg_conceded": 1.05},
    "Universitatea Craiova": {"avg_xg_scored": 1.50, "avg_xg_conceded": 1.15},
    "Rapid București": {"avg_xg_scored": 1.45, "avg_xg_conceded": 1.20},
    "Farul Constanța": {"avg_xg_scored": 1.35, "avg_xg_conceded": 1.30},
    "Sepsi OSK": {"avg_xg_scored": 1.25, "avg_xg_conceded": 1.20},
    "UTA Arad": {"avg_xg_scored": 1.15, "avg_xg_conceded": 1.35},
    "U Cluj": {"avg_xg_scored": 1.20, "avg_xg_conceded": 1.15},
    "Dinamo București": {"avg_xg_scored": 1.20, "avg_xg_conceded": 1.30},
    "FC Botoșani": {"avg_xg_scored": 1.00, "avg_xg_conceded": 1.45},
    "FC Voluntari": {"avg_xg_scored": 1.10, "avg_xg_conceded": 1.25},
    "FC Argeș Pitești": {"avg_xg_scored": 1.05, "avg_xg_conceded": 1.30}
}

@st.cache_data(ttl=600)
def get_league_data(league_code):
    if league_code == "ROU_1":
        return "LOCAL", ROMANIA_FIXTURES

    url = f"https://understat.com/league/{league_code}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        teams_match = re.search(r"teamsData\s*=\s*JSON\.parse\('([^']+)'\)", response.text)
        dates_match = re.search(r"datesData\s*=\s*JSON\.parse\('([^']+)'\)", response.text)
        
        if not teams_match or not dates_match:
            return None, None
            
        teams_json = bytes(teams_match.group(1), 'utf-8').decode('unicode_escape')
        dates_json = bytes(dates_match.group(1), 'utf-8').decode('unicode_escape')
        
        return json.loads(teams_json), json.loads(dates_json)
    except Exception:
        return None, None

def calculate_team_stats(teams_data):
    if teams_data == "LOCAL":
        return ROMANIA_TEAMS, 1.25

    stats = {}
    total_xg_scored, total_xg_conceded, total_games = 0, 0, 0

    for team_id, team in teams_data.items():
        team_name = team['title']
        history = team['history']
        played_games = len(history)
        if played_games == 0:
            continue
            
        xg_scored = sum(float(m['xG']) for m in history)
        xg_conceded = sum(float(m['xGA']) for m in history)
        
        stats[team_name] = {
            "avg_xg_scored": xg_scored / played_games,
            "avg_xg_conceded": xg_conceded / played_games
        }
        total_xg_scored += xg_scored
        total_xg_conceded += xg_conceded
        total_games += played_games

    league_avg_xg = (total_xg_scored / total_games) if total_games > 0 else 1.35
    return stats, league_avg_xg

def calculate_match_probabilities(home_team, away_team, stats, league_avg):
    home_attack = stats[home_team]["avg_xg_scored"] / league_avg
    home_defense = stats[home_team]["avg_xg_conceded"] / league_avg
    
    away_attack = stats[away_team]["avg_xg_scored"] / league_avg
    away_defense = stats[away_team]["avg_xg_conceded"] / league_avg
    
    exp_home_goals = home_attack * away_defense * league_avg * 1.10
    exp_away_goals = away_attack * home_defense * league_avg
    
    max_goals = 8
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = poisson.pmf(i, exp_home_goals) * poisson.pmf(j, exp_away_goals)
            
    prob_home = np.sum(np.tril(matrix, -1))
    prob_draw = np.sum(np.diag(matrix))
    prob_away = np.sum(np.triu(matrix, 1))
    
    return exp_home_goals, exp_away_goals, prob_home, prob_draw, prob_away, matrix

# --- INTERFAȚĂ ---

col_l, col_m = st.columns([1, 2])

with col_l:
    selected_league_label = st.selectbox("Alege Liga:", list(LEAGUES.keys()))
    league_code = LEAGUES[selected_league_label]

    teams_data, dates_data = get_league_data(league_code)

    if teams_data:
        stats, league_avg = calculate_team_stats(teams_data)
        
        st.subheader("📌 Selectează Meciul")
        
        match_options = []
        parsed_matches = []

        if league_code == "ROU_1":
            for m in dates_data:
                match_options.append(f"[{m['status']}] {m['h']} vs {m['a']} ({m['time']})")
                parsed_matches.append((m['h'], m['a']))
        elif dates_data:
            # Extragere meciuri din ultimele 20 de evenimente (inclusiv meciurile de azi)
            recent_and_upcoming = dates_data[-20:]
            
            for m in recent_and_upcoming:
                dt = m.get('datetime', '')[:16].replace(' ', ' ora ')
                is_done = "JUCAT" if m.get('isResult') else "PROGRAMAT/LIVE"
                
                match_options.append(f"[{is_done}] {m['h']['title']} vs {m['a']['title']} ({dt})")
                parsed_matches.append((m['h']['title'], m['a']['title']))

        if match_options:
            selected_match_str = st.selectbox("Meciuri Curente / Programate:", match_options, index=len(match_options)-1 if len(match_options)>0 else 0)
            idx = match_options.index(selected_match_str)
            home_team, away_team = parsed_matches[idx]
        else:
            team_list = sorted(list(stats.keys()))
            home_team = st.selectbox("Echipa Gazdă:", team_list, index=0)
            away_team = st.selectbox("Echipa Oaspete:", team_list, index=min(1, len(team_list)-1))

with col_m:
    if teams_data and stats and 'home_team' in locals() and home_team != away_team:
        exp_home, exp_away, p_home, p_draw, p_away, matrix = calculate_match_probabilities(
            home_team, away_team, stats, league_avg
        )
        
        st.header(f"{home_team} vs {away_team}")
        
        c1, c2 = st.columns(2)
        c1.metric(f"xG Gazde ({home_team})", f"{exp_home:.2f}")
        c2.metric(f"xG Oaspeți ({away_team})", f"{exp_away:.2f}")
        
        st.subheader("📊 Probabilități Rezultat Final (1X2)")
        col_1, col_x, col_2 = st.columns(3)
        col_1.metric("1 (Gazde)", f"{p_home*100:.1f}%", f"Cotă Reală: {1/p_home:.2f}" if p_home > 0 else "")
        col_x.metric("X (Egal)", f"{p_draw*100:.1f}%", f"Cotă Reală: {1/p_draw:.2f}" if p_draw > 0 else "")
        col_2.metric("2 (Oaspeți)", f"{p_away*100:.1f}%", f"Cotă Reală: {1/p_away:.2f}" if p_away > 0 else "")
        
        st.subheader("⚽ Goluri Totale (Peste/Sub 2.5)")
        over_25_prob = 1 - sum(matrix[i, j] for i in range(3) for j in range(3) if i + j <= 2)
        under_25_prob = 1 - over_25_prob
        
        g1, g2 = st.columns(2)
        g1.metric("Peste 2.5 Goluri", f"{over_25_prob*100:.1f}%", f"Cotă: {1/over_25_prob:.2f}" if over_25_prob > 0 else "")
        g2.metric("Sub 2.5 Goluri", f"{under_25_prob*100:.1f}%", f"Cotă: {1/under_25_prob:.2f}" if under_25_prob > 0 else "")
