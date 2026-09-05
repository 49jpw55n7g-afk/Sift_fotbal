import numpy as np
from scipy.stats import poisson

def get_best_match_recommendation(home_xg, away_xg, home_conceded_xg, away_conceded_xg, avg_corners_home, avg_corners_away):
    """
    Analizeaza intreaga piata de pariere pentru un meci din Top 5 Europa
    si returneaza varianta cu cea mai mare probabilitate de reusita.
    """
    # 1. Calcul goluri asteptate (Model Poisson)
    lambda_home = (home_xg + away_conceded_xg) / 2
    lambda_away = (away_xg + home_conceded_xg) / 2
    
    max_g = 6
    matrix = np.zeros((max_g, max_g))
    for h in range(max_g):
        for a in range(max_g):
            matrix[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

    # 2. Piata 1X2 & Sansa Dubla
    p_home = np.sum(np.tril(matrix, -1))
    p_draw = np.sum(np.diag(matrix))
    p_away = np.sum(np.triu(matrix, 1))
    
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw

    # 3. Piata Goluri
    p_over_1_5 = 1 - (matrix[0,0] + matrix[1,0] + matrix[0,1])
    p_under_3_5 = np.sum([matrix[h, a] for h in range(max_g) for a in range(max_g) if h + a < 3.5])
    p_btts = 1 - (np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0, 0])

    # 4. Piata Pauza / Final (Ex: DNB - Draw No Bet)
    p_dnb_1 = p_home / (p_home + p_away) if (p_home + p_away) > 0 else 0

    # 5. Piata Cornere (Model bazat pe medie si distributie)
    exp_corners = avg_corners_home + avg_corners_away
    p_over_7_5_corners = 1 - poisson.cdf(7.5, exp_corners)

    # Centralizam toate optiunile posibile din piata
    market_evaluations = {
        "Sansa Dubla 1X": p_1x,
        "Sansa Dubla X2": p_x2,
        "Peste 1.5 Goluri": p_over_1_5,
        "Sub 3.5 Goluri": p_under_3_5,
        "Ambele Marcheaza (GG)": p_btts,
        "1 Solist (Victorie Gazde)": p_home,
        "2 Solist (Victorie Oaspeti)": p_away,
        "Draw No Bet 1 (Egalul se ramburseaza)": p_dnb_1,
        "Peste 7.5 Cornere": p_over_7_5_corners
    }

    # Filtrare si selectie: alegem pronosticul cu probabilitatea maxima
    best_option = max(market_evaluations, key=market_evaluations.get)
    highest_prob = market_evaluations[best_option]

    # Prag de siguranta: Daca cea mai buna optiune nu trece de 75%, anulam meciul
    if highest_prob >= 0.75:
        return {
            "Status": "RECOMANDAT",
            "Cea mai sigura alegere": best_option,
            "Probabilitate estimata": f"{round(highest_prob * 100, 1)}%",
            "Toate cotele estimate": market_evaluations
        }
    else:
        return {
            "Status": "NO_BET",
            "Motiv": "Meci prea echilibrat (Nicio optiune din piata nu depaseste 75% probabilitate)"
        }
