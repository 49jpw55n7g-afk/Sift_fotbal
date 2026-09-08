import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import poisson
from datetime import datetime, timezone, timedelta

# ============================================================
# 1. CONFIGURARE PAGINĂ & GESTIONARE CHEIE API
# ============================================================

st.set_page_config(
    page_title="Quantum Analytics Engine - Ultra",
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
# 2. HELPERE ȘI ENGINE MATEMATIC EXTINS
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
            return 1.0, 1.0
            
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

    return clamp(l_home, 0.2, 4.5), clamp(l_away, 0.2, 4.5)

def build_dixon_coles_matrix(l_home, l_away):
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            matrix[h, a] = poisson.pmf(h, l_home) * poisson.pmf(a, l_away)
    total = np.sum(matrix)
    return matrix / total if total > 0 else matrix

def extract_all_markets(matrix, l_home, l_away):
    size = matrix.shape[0]
    
    # 1. Piețe Principale 1X2 & Șansă Dublă
    p_1 = float(np.sum(np.tril(matrix, -1)))
    p_x = float(np.sum(np.diag(matrix)))
    p_2 = float(np.sum(np.triu(matrix, 1)))
    p_btts = float(np.sum(matrix[1:, 1:]))

    # 2. Combouri 1X2 + GG / NG
    p_1_gg = float(sum(matrix[h, a] for h in range(1, size) for a in range(1, size) if h > a))
    p_2_gg = float(sum(matrix[h, a] for h in range(1, size) for a in range(1, size) if a > h))
    p_x_gg = float(sum(matrix[i, i] for i in range(1, size)))

    markets = {
        # Soliști
        "1 (Gazde)": p_1,
        "X (Egal)": p_x,
        "2 (Oaspeți)": p_2,
        "1X (Șansă Dublă)": p_1 + p_x,
        "X2 (Șansă Dublă)": p_x + p_2,
        "12 (Fără Egal)": p_1 + p_2,
        
        # Ambele Marchează (GG)
        "GG (Ambele Marchează)": p_btts,
        "NG (Nu Marchează Ambele)": 1.0 - p_btts,
        
        # Combouri
        "1 & GG": p_1_gg,
        "2 & GG": p_2_gg,
        "X & GG": p_x_gg,
        "1 & NG": p_1 - p_1_gg,
        "2 & NG": p_2 - p_2_gg,
        
        # Draw No Bet
        "DNB Gazde": p_1 / (p_1 + p_2) if (p_1 + p_2) > 0 else 0.5,
        "DNB Oaspeți": p_2 / (p_1 + p_2) if (p_1 + p_2) > 0 else 0.5,
    }

    # 3. Linii Total Goluri Meci (Peste / Sub)
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        over_p = sum(matrix[h, a] for h in range(size) for a in range(size) if h + a > line)
        markets[f"Peste {line} Goluri"] = float(over_p)
        markets[f"Sub {line} Goluri"] = float(1.0 - over_p)

    # 4. Total Goluri Echipa Gazdă / Oaspeți
    for line in [0.5, 1.5, 2.5]:
        p_h_over = sum(matrix[h, a] for h in range(size) for a in range(size) if h > line)
        p_a_over = sum(matrix[h, a] for h in range(size) for a in range(size) if a > line)
        markets[f"Gazde Peste {line} Goluri"] = float(p_h_over)
        markets[f"Gazde Sub {line} Goluri"] = float(1.0 - p_h_over)
        markets[f"Oaspeți Peste {line} Goluri"] = float(p_a_over)
        markets[f"Oaspeți Sub {line} Goluri"] = float(1.0 - p_a_over)

    # 5. Handicap European & Asiatic Simplu
    markets["Gazde -1.5"] = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h - a > 1.5))
    markets["Oaspeți +1.5"] = float(sum(matrix[h, a] for h in range(size) for a in range(size) if a - h < 1.5))
    markets["Oaspeți -1.5"] = float(sum(matrix[h, a] for h in range(size) for a in range(size) if a - h > 1.5))
    markets["Gazde +1.5"] = float(sum(matrix[h, a] for h in range(size) for a in range(size) if h - a < 1.5))

    # 6. Estimări Repriza 1 (Aproximare pe distribuție 45% goluri în R1)
    l_h_r1, l_a_r1 = l_home * 0.45, l_away * 0.45
    p_r1_over15 = 1.0 - (poisson.pmf(0, l_h_r1+l_a_r1) + poisson.pmf(1, l_h_r1+l_a_r1))
    p_r1_over05 = 1.0 - poisson.pmf(0, l_h_r1+l_a_r1)
    
    markets["Repriza 1 - Peste 0.5 Goluri"] = float(p_r1_over05)
    markets["Repriza 1 - Peste 1.5 Goluri"] = float(p_r1_over15)
    markets["Repriza 1 - Sub 1.5 Goluri"] = float(1.0 - p_r1_over15)

    return markets

def get_top_correct_scores(matrix, top_n=5):
    scores = []
    for h in range(6):
        for a in range(6):
            p = float(matrix[h, a])
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
    h_goals = top_score["h_goals"]
    a_goals = top_score["a_goals"]
    both_scored = (h_goals > 0 and a_goals > 0)
    total_expected_goals = h_goals + a_goals

    candidates = []
    for market, prob in markets.items():
        # Filtru de consistență logică
        if both_scored and "NG" in market:
            continue
        if not both_scored and market == "GG (Ambele Marchează)":
            continue
            
        if total_expected_goals < 3 and "Peste 2.5" in market:
            continue
        if total_expected_goals >= 3 and "Sub 2.5" in market:
            continue

        odds = fair_odds(prob)
        
        # Filtru de cotă sustenabilă
        if prob >= 0.42 and (1.38 <= odds <= 2.45):
            penalty = 0.90 if "Goluri" in market and "Peste 0.5" not in market else 1.0
            value_score = prob * (odds ** 1.05) * penalty
            candidates.append((market, prob, odds, value_score))

    if candidates:
        best = max(candidates, key=lambda x: x[3])
        return best[0], best[1], best[2]
    else:
        solist_mkts = {k: v for k, v in markets.items() if k in ["1 (Gazde)", "2 (Oaspeți)", "1X (Șansă Dublă)", "X2 (Șansă Dublă)", "GG (Ambele Marchează)"]}
        best_market, best_prob = max(solist_mkts.items(), key=lambda x: x[1])
        return best_market, best_prob, fair_odds(best_prob)

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
                comp_name = item.get("league", {}).get("name", "Competiție")
                
                upcoming.append({
                    "id": item.get("fixture", {}).get("id"),
                    "league_id": league_id,
                    "league": comp_name,
                    "home": item.get("teams", {}).get("home", {}).get("name", "Gazde"),
                    "away": item.get("teams", {}).get("away", {}).get("name", "Oaspeți"),
                    "date": str(item.get("fixture", {}).get("date", today_str))[:10]
                })
    except Exception:
        pass

    for comp_id in [2, 39, 140, 135, 78]:
        try:
            url_hist = f"https://v3.football.api-sports.io/fixtures?league={comp_id}&season={season_year}&last=25"
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

st.title("⚡ Quantum Analytics Engine - Ultra")

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
    with st.spinner("🔄 Se descarcă meciurile și se construiesc modelele statistice..."):
        historical_matches, upcoming_matches = fetch_data_api_sports(st.session_state["api_key"], season_input)

    db, avg_home, avg_away = build_real_team_database(historical_matches)
    
    if not upcoming_matches:
        st.error("❌ Nu s-au primit date de la API. Verifică dacă cheia API este corectă.")
    else:
        leagues_found = list(set([m["league"] for m in upcoming_matches]))
        selected_league_filter = st.sidebar.multiselect("Filtrează după competiție:", options=leagues_found, default=[l for l in leagues_found if "Champions League" in l] or leagues_found)

        filtered_upcoming = [m for m in upcoming_matches if m["league"] in selected_league_filter]

        if not filtered_upcoming:
            st.warning("⚠️ Nu există meciuri pentru competiția selectată.")
        else:
            match_map = {f"[{m['league']}] {m['home']} vs {m['away']}": m for m in filtered_upcoming}
            selected_labels = st.multiselect("Meciuri selectate pentru analiză:", options=list(match_map.keys()), default=list(match_map.keys()))

            selected_matches = [match_map[lbl] for lbl in selected_labels]
            analyzed_matches, ticket_candidates = [], []

            for m in selected_matches:
                l_h, l_a = calculate_expected_goals(m["home"], m["away"], db, avg_home, avg_away)
                matrix = build_dixon_coles_matrix(l_h, l_a)
                mkts = extract_all_markets(matrix, l_h, l_a)
                top_scores = get_top_correct_scores(matrix, top_n=5)

                opt_market, opt_prob, opt_odds = get_best_value_pick(mkts, top_scores[0])

                clean_top_scores = []
                for ts in top_scores:
                    clean_top_scores.append({
                        "Scor Corect": ts["Scor Corect"],
                        "Probabilitate": ts["Probabilitate"],
                        "Cotă Fair": ts["Cotă Fair"]
                    })

                analyzed_matches.append({
                    "match": m,
                    "l_h": l_h,
                    "l_a": l_a,
                    "markets": mkts,
                    "top_scores": clean_top_scores,
                    "opt_market": opt_market,
                    "opt_prob": opt_prob,
                    "opt_odds": opt_odds
                })
                ticket_candidates.append({
                    "Competiție": m["league"],
                    "Meci": f"{m['home']} vs {m['away']}",
                    "Pariu Value": opt_market,
                    "Scor Cel Mai Probabil": clean_top_scores[0]["Scor Corect"],
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
                
                with st.expander(f"⚽ [{m['league']}] {m['home']} vs {m['away']} | xG: {l_h:.2f} - {l_a:.2f}", expanded=True):
                    
                    st.success(f"🔥 **Pariu Optimal Value:** `{opt_m}` | **Probabilitate:** `{opt_p*100:.1f}%` | **Cotă Estimată:** `{opt_o:.2f}`")
                    
                    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
                    
                    with c1:
                        st.markdown("**1X2, DNB & Combouri**")
                        items_c1 = [k for k in mkts.keys() if "Goluri" not in k and "Gazde" not in k and "Oaspeți" not in k and "Repriza" not in k]
                        st.dataframe(pd.DataFrame([{"Piață": k, "Prob.": f"{mkts[k]*100:.1f}%", "Cotă": f"{fair_odds(mkts[k]):.2f}"} for k in items_c1]), use_container_width=True)
                    
                    with c2:
                        st.markdown("**Linii Total Goluri Meci**")
                        items_c2 = [k for k in mkts.keys() if "Goluri" in k and "Gazde" not in k and "Oaspeți" not in k and "Repriza" not in k]
                        st.dataframe(pd.DataFrame([{"Piață": k, "Prob.": f"{mkts[k]*100:.1f}%", "Cotă": f"{fair_odds(mkts[k]):.2f}"} for k in items_c2]), use_container_width=True)

                    with c3:
                        st.markdown("**Goluri Echipă & Reprize**")
                        items_c3 = [k for k in mkts.keys() if "Gazde" in k or "Oaspeți" in k or "Repriza" in k]
                        st.dataframe(pd.DataFrame([{"Piață": k, "Prob.": f"{mkts[k]*100:.1f}%", "Cotă": f"{fair_odds(mkts[k]):.2f}"} for k in items_c3]), use_container_width=True)

                    with c4:
                        st.markdown("**🎯 Top 5 Scoruri Corecte**")
                        st.dataframe(pd.DataFrame(top_scores), use_container_width=True)
