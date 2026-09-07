import csv
import os
from understatapi import Understat


def resolve_pending_bets(filepath: str = "history_value_bets.csv", season: int = 2025):
    if not os.path.exists(filepath):
        print("Nu există niciun fișier de istoric CSV.")
        return

    understat = Understat()
    rows = []
    total_profit = 0.0
    total_staked = 0.0

    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = list(csv.reader(file))
        if not reader:
            return
        header = reader[0]
        data = reader[1:]

    updated = False
    for row in data:
        if len(row) < 10:
            rows.append(row)
            continue

        status = row[8]
        if status == "PENDING":
            match_name = row[2]
            home_team, away_team = match_name.split(" vs ")
            market = row[3]
            odds = float(row[4])
            stake_pct = float(row[7].replace("% din bancă", ""))

            try:
                results = understat.get_team_results(team_name=home_team, season=season)
                # Căutăm meciul jucat
                match_data = next((m for m in results if m['h']['title'] == home_team and m['a']['title'] == away_team and m.get('goals') is not None), None)

                if match_data:
                    h_goals = int(match_data['goals']['h'])
                    a_goals = int(match_data['goals']['a'])
                    won = False

                    # Verificare condiții de câștig
                    if market == "1 Solist" and h_goals > a_goals: won = True
                    elif market == "2 Solist" and a_goals > h_goals: won = True
                    elif market == "Egal (X)" and h_goals == a_goals: won = True
                    elif market == "Sansa Dubla 1X" and h_goals >= a_goals: won = True
                    elif market == "Sansa Dubla X2" and a_goals >= h_goals: won = True
                    elif market == "Peste 1.5 Goluri" and (h_goals + a_goals) > 1.5: won = True
                    elif market == "Peste 2.5 Goluri" and (h_goals + a_goals) > 2.5: won = True
                    elif market == "Sub 3.5 Goluri" and (h_goals + a_goals) < 3.5: won = True
                    elif market == "Ambele Marcheaza (GG)" and h_goals > 0 and a_goals > 0: won = True

                    profit = round((odds - 1.0) * stake_pct, 2) if won else -stake_pct
                    row[8] = "WON" if won else "LOST"
                    row[9] = str(profit)
                    updated = True
            except Exception as e:
                print(f"Eroare verificare meci {match_name}: {e}")

        # Calcul statistici globale
        if row[8] in ["WON", "LOST"]:
            stake = float(row[7].replace("% din bancă", ""))
            profit = float(row[9])
            total_staked += stake
            total_profit += profit

        rows.append(row)

    if updated:
        with open(filepath, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        print("✅ Istoric actualizat cu rezultatele recente!")

    if total_staked > 0:
        roi = round((total_profit / total_staked) * 100, 2)
        print(f"📊 PERFORMANȚĂ TOTALĂ: Profit Total = {round(total_profit, 2)}% | ROI = {roi}%")


if __name__ == "__main__":
    resolve_pending_bets()
