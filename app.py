import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import poisson
from datetime import datetime, timezone

# ============================================================
# 1. CONFIGURARE PAGINĂ
# ============================================================

st.set_page_config(
    page_title="Quantum Live & Pre-Match Engine",
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

# ============================================================
# 2. ENGINE MATEMATIC CORECTAT
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
    
    base_gf = np.random.uniform(0.9, 2.1)
    base_ga = np.random.uniform(0.8, 1.8)
    
    history = []
    for i in range(15):
        gf = max(0, int(np.random.poisson(base_gf)))
        ga = max(0, int(np.random.poisson(base_ga)))
        history.append({
            "venue": "home" if i % 2 == 0 else "away",
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
    """Construiește matricea adăugând scorul deja existent dacă meciul e LIVE."""
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

def get_best_value_pick(markets):
    """
    Selectează opțiunea optimă în mod matematic STRICT, fără favoritism.
    Alege varianta cu cel mai bun raport între Probabilitate și Cotă.
    """
    candidates = []
    
    for market, prob in markets.items():
        odds = fair_odds(prob)
        
        # Filtru: căutăm pariuri de siguranță/valoare
        if prob >= 0.45 and (1.25 <= odds <= 2.80):
            # Prioritizăm piețele cu șansă reală de câștig
            score = prob * (odds ** 0.8)
            candidates.append((market, prob, odds, score))
            
    if candidates:
        best = max(candidates, key=lambda x: x[3])
        return best[0], best[1], best[2]
        
    # Fallback pe cea mai probabilă opțiune de șansă dublă
    if markets["X2 (Șansă Dublă)"] >= markets["1X (Șansă Dublă)"]:
        m = "X2 (Șansă Dublă)"
    else:
        m = "1X (Șansă Dublă)"
    return m, markets[m], fair_odds(markets[m])

# ============================================================
# 3. PRELUARE DATE API (CU SUPORT LIVE)
# ============================================================

@st.cache_data(ttl=60)
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
                status_short = item.get("fixture", {}).get("status", {}).get("short")
                is_live = status_short in ["1H", "HT", "2H", "ET", "P"]
                
                sh = item.get("goals", {}).get("home") if is_live else 0
                sa = item.get("goals", {}).get("away") if is_live else 0
                
                upcoming.append({
                    "id": item.get("fixture", {}).get("id"),
                    "league": item.get("league", {}).get("name", "Competiție"),
                    "home": item.get("teams", {}).get("home", {}).get("name", "Gazde"),
                    "away": item.get("teams", {}).get("away", {}).get("name", "Oaspeți"),
                    "is_live": is_live,
                    "live_home": sh if sh is not None else 0,
                    "live_away": sa if sa is not None else 0,
                    "status": status_short,
                    "elapsed": item.get("fixture", {}).get("status", {}).get("elapsed", 0),
                    "date": str(item.get("fixture", {}).get("date", today_str))[:10]
                })
    except Exception:
        pass

    for comp_id in [2, 39, 140, 135, 78]:
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
# 4. INTERFAȚĂ
# ============================================================

st.title("⚡ Quantum Live & Pre-Match Engine")

with st.sidebar:
    st.header("⚙️ Setări")
    input_key = st.text_input("🔑 Cheie API:", value=st.session_state["api_key"], type="password")
    if st.button("✅ Salvează"):
        st.session_state["api_key"] = input_key.strip()
        st.rerun()
    season_input = st.number_input("📅 Sezon:", min_value=2023, max_value=2026, value=2026)

if not st.session_state["api_key"]:
    st.info("👈 Introdu Cheia API în meniul lateral.")
else:
    with st.spinner("🔄 Se actualizează datele în timp real..."):
        historical_matches, upcoming_matches = fetch_data_api_sports(st.session_state["api_key"], season_input)

    db, avg_home, avg_away = build_real_team_database(historical_matches)

    if not upcoming_matches:
        st.error("❌ Nu s-au găsit meciuri.")
    else:
        match_map = {
            f"{'🔴 LIVE ' + str(m['live_home']) + '-' + str(m['live_away']) + ' ' if m['is_live'] else ''}[{m['league']}] {m['home']} vs {m['away']}": m 
            for m in upcoming_matches
        }
        
        selected_labels = st.multiselect("Meciuri selectate:", options=list(match_map.keys()), default=list(match_map.keys())[:5])
        selected_matches = [match_map[lbl] for lbl in selected_labels]

        analyzed_matches = []
        for m in selected_matches:
            l_h, l_a = calculate_expected_goals(m["home"], m["away"], db, avg_home, avg_away)
            
            # Ajustăm xG-ul rămas dacă meciul e LIVE în funcție de minut
            if m["is_live"]:
                time_remaining = max(5, 90 - (m["elapsed"] or 45)) / 90.0
                l_h *= time_remaining
                l_a *= time_remaining
                matrix = build_dixon_coles_matrix(l_h, l_a, base_h=m["live_home"], base_a=m["live_away"])
            else:
                matrix = build_dixon_coles_matrix(l_h, l_a)

            mkts = extract_all_markets(matrix)
            top_scores = get_top_correct_scores(matrix)
            opt_m, opt_p, opt_o = get_best_value_pick(mkts)

            analyzed_matches.append({
                "match": m,
                "l_h": l_h,
                "l_a": l_a,
                "markets": mkts,
                "top_scores": top_scores,
                "opt_market": opt_m,
                "opt_prob": opt_p,
                "opt_odds": opt_o
            })

        st.markdown("---")
        st.header("📊 Analiză Meciuri & Predicții")

        for item in analyzed_matches:
            m = item["match"]
            mkts = item["markets"]
            top_scores = item["top_scores"]
            opt_m, opt_p, opt_o = item["opt_market"], item["opt_prob"], item["opt_odds"]

            status_tag = f"🔴 LIVE ({m['live_home']}-{m['live_away']}, Min {m['elapsed']}')" if m["is_live"] else "⏳ PRE-MATCH"
            
            with st.expander(f"{status_tag} [{m['league']}] {m['home']} vs {m['away']}", expanded=True):
                st.success(f"🔥 **Pariu Optimal Value:** `{opt_m}` | **Probabilitate:** `{opt_p*100:.1f}%` | **Cotă Estimată:** `{opt_o:.2f}`")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**1X2 & Combo**")
                    items_c1 = [k for k in mkts.keys() if "Goluri" not in k]
                    st.dataframe(pd.DataFrame([{"Piață": k, "Prob.": f"{mkts[k]*100:.1f}%", "Cotă": f"{fair_odds(mkts[k]):.2f}"} for k in items_c1]), use_container_width=True)
                
                with c2:
                    st.markdown("**Linii Goluri**")
                    items_c2 = [k for k in mkts.keys() if "Goluri" in k]
                    st.dataframe(pd.DataFrame([{"Piață": k, "Prob.": f"{mkts[k]*100:.1f}%", "Cotă": f"{fair_odds(mkts[k]):.2f}"} for k in items_c2]), use_container_width=True)

                with c3:
                    st.markdown("**🎯 Top Scoruri Posibile**")
                    st.dataframe(pd.DataFrame([{"Scor": s["Scor Corect"], "Prob.": s["Probabilitate"], "Cotă": s["Cotă Fair"]} for s in top_scores]), use_container_width=True)
