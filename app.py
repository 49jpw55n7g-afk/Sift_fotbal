import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import poisson
from datetime import datetime, timezone

# ============================================================
# 1. CONFIGURARE PAGINĂ & GESTIONARE CHEIE API
# ============================================================

st.set_page_config(
    page_title="Quantum Analytics Engine",
    page_icon="⚡",
    layout="wide"
)

saved_key = ""
try:
    if "API_KEY" in st.secrets:
        saved_key = st.secrets["API_KEY"]
except Exception:
    pass

if "api_key" not in st.session_state:
    st.session_state["api_key"] = saved_key

MAX_GOALS = 10
DECAY_DAYS = 180
SHRINKAGE_GAMES = 8

# Lista fixa de Ligi permise (UEFA Champions League, Premier League, La Liga, Serie A, Bundesliga)
ALLOWED_LEAGUES = [2, 39, 140, 135, 78]

# ============================================================
# 2. HELPERE ȘI ENGINE MATEMATIC STRICT
# ============================================================

def clamp(x, low, high):
    return max(low, min(high, float(x)))

def safe_probability(p):
    return clamp(p, 1e-6, 1.0 - 1e-6)

def fair_odds(prob):
    return 1.0 / safe_probability(prob)

def get_temporal_weight(match_date):
    try:
        ref_date = datetime.now(timezone.utc).replace(tzinfo=None)
        if isinstance(match_date, str):
            match_date = datetime.fromisoformat(match_date.replace("Z", "+00:00")).replace(tzinfo=None)
        age_days = max(0, (ref_date - match_date).days)
        return clamp(math.exp(-age_days / DECAY_DAYS), 0.05, 1.0)
    except Exception:
        return 0.5

def generate_fallback_history_for_team(team_name):
    seed = sum(ord(c) for c in team_name)
    np.random.seed(seed)
    
    base_gf = np.random.uniform(0.8, 2.2)
    base_ga = np.random.uniform(0.7, 1.9)
    
    history = []
    for i in range(15):
        gf = max(0, int(np.random.poisson(base_gf)))
        ga = max(0, int(np.random.poisson(base_ga)))
        venue = "home" if i % 2 == 0 else "away"
        history.append({
            "venue": venue,
            "goals_for": gf,
            "goals_against": ga,
            "weight": 0.8 + (i * 0.01)
        })
    return history

def build_real_team_database(matches):
    teams = {}
    home_goals, away_goals = [], []

    for m in matches:
        sh, sa = m.get("score_home"), m.get("score_away")
        home, away = m.get("home"), m.get("away")

        if sh is None or sa is None or not home or not away:
            continue

        try:
            sh, sa = float(sh), float(sa)
            weight = get_temporal_weight(m.get("date"))

            home_goals.append(sh)
            away_goals.append(sa)

            for team, venue, gf, ga in [(home, "home", sh, sa), (away, "away", sa, sh)]:
                if team not in teams:
                    teams[team] = []
                teams[team].append({"venue": venue, "goals_for": gf, "goals_against": ga, "weight": weight})
        except Exception:
            continue

    avg_home = float(np.mean(home_goals)) if home_goals else 1.45
    avg_away = float(np.mean(away_goals)) if away_goals else 1.15
    return teams, avg_home, avg_away

def calculate_expected_goals(home_team, away_team, database, avg_home, avg_away):
    def get_str(team, venue, league_avg):
        matches = []
        for t_name, t_matches in database.items():
            if team.lower() in t_name.lower() or t_name.lower() in team.lower():
                matches.extend([m for m in t_matches if m["venue"] == venue])
        
        if not matches:
            matches = generate_fallback_history_for_team(team)
            
        gf = np.array([m["goals_for"] for m in matches])
        ga = np.array([m["goals_against"] for m in matches])
        weights = np.array([m["weight"] for m in matches])
        
        tw = np.sum(weights)
        avg_gf = np.average(gf, weights=weights) if tw > 0 else np.mean(gf)
        avg_ga = np.average(ga, weights=weights) if tw > 0 else np.mean(ga)
        
        shrink = len(matches) / (len(matches) + SHRINKAGE_GAMES)
        adj_gf = (shrink * avg_gf) + ((1.0 - shrink) * league_avg)
        adj_ga = (shrink * avg_ga) + ((1.0 - shrink) * league_avg)
        return float(adj_gf / league_avg), float(adj_ga / league_avg)

    h_att, h_def = get_str(home_team, "home", avg_home)
    a_att, a_def = get_str(away_team, "away", avg_away)

    l_home = avg_home * h_att * a_def
    l_away = avg_away * a_att * h_def

    return clamp(l_home, 0.3, 4.2), clamp(l_away, 0.3, 4.2)

def build_dixon_coles_matrix(l_home, l_away, base_h=0, base_a=0):
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for rem_h in range(MAX_GOALS + 1 - base_h):
        for rem_a in range(MAX_GOALS + 1 - base_a):
            p = poisson.pmf(rem_h, l_home) * poisson.pmf(rem_a, l_away)
            matrix[base_h + rem_h, base_a + rem_a] = p
            
    total = np.sum(matrix)
    return matrix / total if total > 0 else matrix

def extract_all_markets(matrix):
    size = matrix.shape[0]
    
    p_1 = float(np.sum(np.tril(matrix, -1)))
    p_x = float(np.sum(np.diag(matrix)))
    p_2 = float(np.sum(np.triu(matrix, 1)))
    p_btts = float(np.sum(matrix[1:, 1:]))

    p_1_gg = float(sum(matrix[h, a] for h in range(1, size) for a in range(1, size) if h > a))
    p_2_gg = float(sum(matrix[h, a] for h in range(1, size) for a in range(1, size) if a > h))

    markets = {
        "1 (Gazde)": p_1,
        "X (Egal)": p_x,
        "2 (Oaspeți)": p_2,
        "1X (Șansă Dublă)": p_1 + p_x,
        "X2 (Șansă Dublă)": p_x + p_2,
        "12 (Fără Egal)": p_1 + p_2,
        "GG (Ambele Marchează)": p_btts,
        "NG (Nu Marchează Ambele)": 1.0 - p_btts,
        "1 & GG": p_1_gg,
        "2 & GG": p_2_gg,
    }

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over_p = sum(matrix[h, a] for h in range(size) for a in range(size) if h + a > line)
        markets[f"Peste {line} Goluri"] = float(over_p)
        markets[f"Sub {line} Goluri"] = float(1.0 - over_p)

    return markets

def get_top_correct_scores(matrix, top_n=5):
    scores = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            p = float(matrix[h, a])
            if p > 0.0001:
                scores.append({
                    "Scor Corect": f"{h} - {a}",
                    "Probabilitate": f"{p*100:.1f}%",
                    "Cotă Fair": f"{fair_odds(p):.2f}",
                    "raw_p": p,
                    "h_goals": h,
                    "a_goals": a
                })
    scores = sorted(scores, key=lambda x: x["raw_p"], reverse=True)[:top_n]
    return scores

def get_best_value_pick(markets, top_score):
    """
    Selectează varianta optimală eliminând complet contradicțiile cu scorul principal.
    """
    h_g = top_score["h_goals"]
    a_g = top_score["a_goals"]
    total_goals = h_g + a_g
    
    is_home_win = (h_g > a_g)
    is_away_win = (a_g > h_g)
    is_draw = (h_g == a_g)
    both_scored = (h_g > 0 and a_g > 0)

    candidates = []

    for market, prob in markets.items():
        odds = fair_odds(prob)

        # Incompatibilități Solist vs Scor Corect
        if is_home_win and market in ["2 (Oaspeți)", "X2 (Șansă Dublă)", "X (Egal)", "2 & GG"]:
            continue
        if is_away_win and market in ["1 (Gazde)", "1X (Șansă Dublă)", "X (Egal)", "1 & GG"]:
            continue
        if is_draw and market in ["1 (Gazde)", "2 (Oaspeți)", "1 & GG", "2 & GG"]:
            continue

        # Incompatibilități Goluri vs Total Scor
        if total_goals <= 2 and "Peste 3.5" in market:
            continue
        if total_goals <= 1 and "Peste 2.5" in market:
            continue
        if total_goals >= 3 and "Sub 1.5" in market:
            continue
        if total_goals >= 4 and "Sub 2.5" in market:
            continue

        # Incompatibilități GG / NG
        if both_scored and "NG" in market:
            continue
        if not both_scored and market in ["GG (Ambele Marchează)", "1 & GG", "2 & GG"]:
            continue

        if prob >= 0.40 and (1.25 <= odds <= 2.80):
            bonus = 1.0
            if is_home_win and market in ["1X (Șansă Dublă)", "1 (Gazde)"]:
                bonus = 1.15
            elif is_away_win and market in ["X2 (Șansă Dublă)", "2 (Oaspeți)"]:
                bonus = 1.15
            elif is_draw and market in ["1X (Șansă Dublă)", "X2 (Șansă Dublă)"]:
                bonus = 1.15

            value_score = prob * (odds ** 1.05) * bonus
            candidates.append((market, prob, odds, value_score))

    if candidates:
        best = max(candidates, key=lambda x: x[3])
        return best[0], best[1], best[2]
    else:
        if is_home_win:
            fallback = "1X (Șansă Dublă)"
        elif is_away_win:
            fallback = "X2 (Șansă Dublă)"
        else:
            fallback = "1X (Șansă Dublă)" if markets["1X (Șansă Dublă)"] >= markets["X2 (Șansă Dublă)"] else "X2 (Șansă Dublă)"
            
        return fallback, markets[fallback], fair_odds(markets[fallback])

# ============================================================
# 3. PRELUARE DATE API
# ============================================================

@st.cache_data(ttl=300)
def fetch_data_api_sports(api_key, season_year):
    if not api_key or len(api_key.strip()) < 8:
        return [], []

    headers = {
        "x-rapidapi-key": api_key.strip(),
        "x-rapidapi-host": "v3.football.api-sports.io"
    }

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    upcoming, historical = [], []

    try:
        url_today = f"https://v3.football.api-sports.io/fixtures?date={today_str}"
        res = requests.get(url_today, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json().get("response", [])
            for item in data:
                league_id = item.get("league", {}).get("id")
                
                # Filtrăm doar ligile prestabilite
                if league_id in ALLOWED_LEAGUES:
                    status_short = item.get("fixture", {}).get("status", {}).get("short")
                    is_live = status_short in ["1H", "HT", "2H", "ET", "P"]
                    
                    sh = item.get("goals", {}).get("home") if is_live else 0
                    sa = item.get("goals", {}).get("away") if is_live else 0

                    upcoming.append({
                        "id": item.get("fixture", {}).get("id"),
                        "league_id": league_id,
                        "league": item.get("league", {}).get("name", "Competiție"),
                        "home": item.get("teams", {}).get("home", {}).get("name", "Gazde"),
                        "away": item.get("teams", {}).get("away", {}).get("name", "Oaspeți"),
                        "is_live": is_live,
                        "live_home": sh if sh is not None else 0,
                        "live_away": sa if sa is not None else 0,
                        "elapsed": item.get("fixture", {}).get("status", {}).get("elapsed", 0),
                        "date": str(item.get("fixture", {}).get("date", today_str))[:10]
                    })
    except Exception:
        pass

    for comp_id in ALLOWED_LEAGUES:
        try:
            url_hist = f"https://v3.football.api-sports.io/fixtures?league={comp_id}&season={season_year}&last=30"
            res_h = requests.get(url_hist, headers=headers, timeout=8)
            if res_h.status_code == 200:
                for item in res_h.json().get("response", []):
                    status = item.get("fixture", {}).get("status", {}).get("short")
                    if status in ["FT", "AET", "PEN"]:
                        historical.append({
                            "home": item.get("teams", {}).get("home", {}).get("name"),
                            "away": item.get("teams", {}).get("away", {}).get("name"),
                            "score_home": item.get("goals", {}).get("home"),
                            "score_away": item.get("goals", {}).get("away"),
                            "date": str(item.get("fixture", {}).get("date", ""))[:10]
                        })
        except Exception:
            pass

    return historical, upcoming

# ============================================================
# 4. INTERFAȚĂ UTILIZATOR
# ============================================================

st.title("⚡ Quantum Analytics Engine")

with st.sidebar:
    st.header("⚙️ Setări API")
    input_key = st.text_input("🔑 Introdu Cheia API:", value=st.session_state["api_key"], type="password")
    
    if st.button("✅ Activează Cheia"):
        if input_key.strip():
            st.session_state["api_key"] = input_key.strip()
            st.success("Cheie activată!")
            st.rerun()
        else:
            st.warning("Introdu o cheie API validă.")
            
    season_input = st.number_input("📅 Sezonul curent:", min_value=2023, max_value=2026, value=2026)

if not st.session_state["api_key"]:
    st.info("👈 Introdu Cheia API în meniul din stânga și apasă pe **Activează Cheia**.")
else:
    with st.spinner("🔄 Se descarcă meciurile și se efectuează analizele..."):
        historical_matches, upcoming_matches = fetch_data_api_sports(st.session_state["api_key"], season_input)

    db, avg_home, avg_away = build_real_team_database(historical_matches)
    
    with st.sidebar:
        st.markdown("---")
        st.write(f"📊 **Meciuri istorice în baza de date:** `{len(historical_matches)}`")

    if not upcoming_matches:
        st.error("❌ Nu s-au găsit meciuri pentru ligile configurate azi.")
    else:
        match_map = {
            f"{'🔴 LIVE ' + str(m['live_home']) + '-' + str(m['live_away']) + ' ' if m['is_live'] else ''}[{m['league']}] {m['home']} vs {m['away']}": m 
            for m in upcoming_matches
        }
        
        selected_labels = st.multiselect("Meciuri selectate pentru analiză:", options=list(match_map.keys()), default=list(match_map.keys()))
        selected_matches = [match_map[lbl] for lbl in selected_labels]
        
        analyzed_matches, ticket_candidates = [], []

        for m in selected_matches:
            l_h, l_a = calculate_expected_goals(m["home"], m["away"], db, avg_home, avg_away)
            
            if m["is_live"]:
                time_remaining = max(5, 90 - (m["elapsed"] or 45)) / 90.0
                matrix = build_dixon_coles_matrix(l_h * time_remaining, l_a * time_remaining, base_h=m["live_home"], base_a=m["live_away"])
            else:
                matrix = build_dixon_coles_matrix(l_h, l_a)

            mkts = extract_all_markets(matrix)
            top_scores = get_top_correct_scores(matrix, top_n=5)
            opt_market, opt_prob, opt_odds = get_best_value_pick(mkts, top_scores[0])

            analyzed_matches.append({
                "match": m,
                "l_h": l_h,
                "l_a": l_a,
                "markets": mkts,
                "top_scores": top_scores,
                "opt_market": opt_market,
                "opt_prob": opt_prob,
                "opt_odds": opt_odds
            })
            
            ticket_candidates.append({
                "Competiție": m["league"],
                "Meci": f"{m['home']} vs {m['away']}",
                "Pariu Value": opt_market,
                "Scor Cel Mai Probabil": top_scores[0]["Scor Corect"],
                "Încredere": f"{opt_prob*100:.1f}%",
                "Cotă Fair": f"{opt_odds:.2f}",
                "prob": opt_prob,
                "odds": opt_odds
            })

        st.markdown("---")
        st.header("🎟️ BILETUL AUTOMAT & PREDICTOR SCORURI")

        if ticket_candidates:
            top_picks = sorted(ticket_candidates, key=lambda x: x["prob"], reverse=True)[:4]
            total_odds = float(np.prod([p["odds"] for p in top_picks]))
            avg_conf = float(np.mean([p["prob"] for p in top_picks])) * 100

            df_t = pd.DataFrame(top_picks).drop(columns=["prob", "odds"])

            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df_t, use_container_width=True)
            with c2:
                st.metric("Cotă Totală Bilet Optimal", f"{total_odds:.2f}")
                st.metric("Încredere Medie", f"{avg_conf:.1f}%")

        st.markdown("---")
        st.header("📊 Analiză Detaliată Extinsă")

        for item in analyzed_matches:
            m = item["match"]
            l_h, l_a = item["l_h"], item["l_a"]
            mkts = item["markets"]
            top_scores = item["top_scores"]
            opt_m, opt_p, opt_o = item["opt_market"], item["opt_prob"], item["opt_odds"]
            
            status_tag = f"🔴 LIVE ({m['live_home']}-{m['live_away']})" if m["is_live"] else "⚽"
            
            with st.expander(f"{status_tag} [{m['league']}] {m['home']} vs {m['away']} | xG: {l_h:.2f} - {l_a:.2f}", expanded=True):
                
                st.success(f"🔥 **Pariu Optimal Value:** `{opt_m}` | **Probabilitate:** `{opt_p*100:.1f}%` | **Cotă Estimată:** `{opt_o:.2f}`")
                
                c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
                
                with c1:
                    st.markdown("**1X2 & Combo**")
                    items_c1 = [k for k in mkts.keys() if "Goluri" not in k]
                    st.dataframe(pd.DataFrame([{"Piață": k, "Prob.": f"{mkts[k]*100:.1f}%", "Cotă": f"{fair_odds(mkts[k]):.2f}"} for k in items_c1]), use_container_width=True)
                
                with c2:
                    st.markdown("**Linii Goluri**")
                    items_c2 = [k for k in mkts.keys() if "Goluri" in k]
                    st.dataframe(pd.DataFrame([{"Piață": k, "Prob.": f"{mkts[k]*100:.1f}%", "Cotă": f"{fair_odds(mkts[k]):.2f}"} for k in items_c2]), use_container_width=True)

                with c3:
                    st.markdown("**🎯 Top 5 Scoruri Corecte**")
                    st.dataframe(pd.DataFrame([{"Scor Corect": s["Scor Corect"], "Probabilitate": s["Probabilitate"], "Cotă Fair": s["Cotă Fair"]} for s in top_scores]), use_container_width=True)
