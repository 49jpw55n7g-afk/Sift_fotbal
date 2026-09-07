import json
import re
from datetime import datetime
import requests
import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Predicții xG Meciuri", page_icon="⚽", layout="wide")

st.title("⚽ Predicții Automate xG - Meciuri & Analiză")

LEAGUES = {
    "🇮🇹 Serie A (Italia)": "Serie_A",
    "🇪🇸 La Liga (Spania)": "La_Liga",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Anglia)": "EPL",
    "🇩🇪 Bundesliga (Germania)": "Bundesliga",
    "🇫🇷 Ligue 1 (Franța)": "Ligue_1",
    "🇷🇴 Superliga (România)": "ROU_1"
}

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
    "Oțelul Galați": {"avg_xg_scored": 1.10, "avg_xg_conceded": 1.05},
    "Petrolul Ploiești": {"avg_xg_scored": 1.05, "avg_xg_conceded": 1.25},
    "FC Hermannstadt": {"avg_xg_scored": 1.15, "avg_xg_conceded": 1.25}
}

@st.cache_data(ttl=1800)
def get_understat_data(league_code):
    if league_code == "ROU_1":
        return "LOCAL", None

    url = f"https://understat.com/league/{league_code}"
    headers = {"User-Agent": "Mozilla/5.0"}
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

def find_best_team_match(name, stats):
    if name in stats:
        return name
    # Căutare după potrivire parțială de nume
    for team in stats.keys():
        if name.lower() in team.lower() or team.lower() in name.lower():
            return team
    return list(stats.keys())[0]

def calculate_match_probabilities(home_team, away_team, stats, league_avg):
    home_key = find_best_team_match(home_team, stats)
    away_key = find_best_team_match(away_team, stats)

    h_stat = stats.get(home_key, {"avg_xg_scored": 1.20, "avg_xg_conceded": 1.20})
    a_stat = stats.get(away_key, {"avg_xg_scored": 1.20, "avg_xg_conceded": 1.20})

    home_attack = h_stat["avg_xg_scored"] / league_avg
    home_defense = h_stat["avg_xg_conceded"] / league_avg
    away_attack = a_stat["avg_xg_scored"] / league_avg
    away_defense = a_stat["avg_xg_conceded"] / league_avg
    
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
    
    return home_key, away_key, exp_home_goals, exp_away_goals, prob_home, prob_draw, prob_away, matrix

# --- INTERFAȚĂ ---

col_l, col_m = st.columns([1, 2])

with col_l:
    selected_league_label = st.selectbox("Alege Liga:", list(LEAGUES.keys()))
    league_code = LEAGUES[selected_league_label]

    teams_data, dates_data = get_understat_data(league_code)

    if teams_data:
        stats, league_avg = calculate_team_stats(teams_data)
        
        st.subheader("📌 Meciuri Disponibile")
        
        match_options = []
        parsed_matches = []

        if league_code == "ROU_1":
            team_list = sorted(list(stats.keys()))
            st.info("Alege meciul manual sau modifică echipele:")
            home_team = st.selectbox("Echipa Gazdă:", team_list, index=0)
            away_team = st.selectbox("Echipa Oaspete:", team_list, index=min(1, len(team_list)-1))
        elif dates_data:
            filter_type = st.radio("Filtrare Meciuri:", ["Următoarele Meciuri / Etapa Curentă", "Ultimele Rezultate"], index=0)
            
            if filter_type == "Următoarele Meciuri / Etapa Curentă":
                upcoming = [m for m in dates_data if not m.get('isResult')]
                if not upcoming:
                    st.warning("Nu există meciuri nejucte în perioada imediat următoare. Afișăm ultimele meciuri:")
                    matches_to_show = dates_data[-10:]
                else:
                    matches_to_show = upcoming[:10]
            else:
                matches_to_show = [m for m in dates_data if m.get('isResult')][-12:]
                matches_to_show.reverse()

            for m in matches_to_show:
                dt = m.get('datetime', '')[:16].replace(' ', ' - ')
                status = "JUCAT" if m.get('isResult') else "PROGRAMAT"
                
                h_name = m['h']['title']
                a_name = m['a']['title']
                
                match_options.append(f"[{status}] {h_name} vs {a_name} ({dt})")
                parsed_matches.append((h_name, a_name))

            if match_options:
                selected_str = st.selectbox("Selectează Meciul:", match_options)
                idx = match_options.index(selected_str)
                home_team, away_team = parsed_matches[idx]

with col_m:
    if teams_data and stats and 'home_team' in locals() and home_team != away_team:
        h_matched, a_matched, exp_home, exp_away, p_home, p_draw, p_away, matrix = calculate_match_probabilities(
            home_team, away_team, stats, league_avg
        )
        
        st.header(f"{h_matched} vs {a_matched}")
        
        c1, c2 = st.columns(2)
        c1.metric(f"xG Gazde ({h_matched})", f"{exp_home:.2f}")
        c2.metric(f"xG Oaspeți ({a_matched})", f"{exp_away:.2f}")
        
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
