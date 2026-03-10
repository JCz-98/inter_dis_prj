# import math
# import os
# import subprocess
# from collections import defaultdict

# import pandas as pd
# import trueskill

# # --- Parameters ---
# BASE_ELO = 1000
# BASE_K = 32
# HIGH_K = 48
# LOW_K = 16
# EXPERIENCE_THRESHOLD = 100

# # --- TrueSkill Setup ---
# # draw_probability=0.0 enforces a strict Win/Loss environment
# ts_env = trueskill.TrueSkill(draw_probability=0.0, mu=25.0, sigma=8.333)


# def get_k_factor(games_played, overtime):
#     """
#     Dynamic K-factor:
#     - High volatility for new players (games < 100)
#     - Low volatility for veterans
#     - Halved volatility for Overtime (closer games = less rating swap)
#     """
#     base_k = HIGH_K if games_played < EXPERIENCE_THRESHOLD else LOW_K
#     if overtime:
#         return base_k / 2
#     return base_k


# def calculate_elo_change(r_player, r_opponent_avg, actual_score, k, margin):
#     """Standard Elo update formula with Margin of Victory multiplier."""
#     expected = 1 / (1 + 10 ** ((r_opponent_avg - r_player) / 400))
#     return k * margin * (actual_score - expected)


# def calculate_trueskill_with_margin(
#     team_red, team_blue,
#     score_red, score_blue,
#     is_overtime=False,
#     env=None
# ):
#     """
#     Adapts TrueSkill update to incorporate score magnitude and overtime.

#     Args:
#         team_red (list): List of trueskill.Rating objects for Red team.
#         team_blue (list): List of trueskill.Rating objects for Blue team.
#         score_red (int): Red team score.
#         score_blue (int): Blue team score.
#         is_overtime (bool): Was the match decided in OT?
#         env (trueskill.TrueSkill): Optional custom environment.

#     Returns:
#         (new_red_team, new_blue_team): Tuple of lists with updated Rating objects.
#     """
#     if env is None:
#         env = trueskill.TrueSkill()

#     # 1. Determine Winner and basic Ranks
#     if score_red > score_blue:
#         ranks = [0, 1]  # Red wins
#         winner_team, loser_team = team_red, team_blue
#     elif score_blue > score_red:
#         ranks = [1, 0]  # Blue wins
#         winner_team, loser_team = team_blue, team_red
#     else:
#         # Draw (if your sport allows it)
#         ranks = [0, 0]
#         return env.rate([team_red, team_blue], ranks=ranks)

#     # 2. Calculate the "Standard" TrueSkill Update (Binary Win/Loss)
#     # We group them to get the raw calculated updates
#     rating_groups = [team_red, team_blue]
#     new_ratings_groups = env.rate(rating_groups, ranks=ranks)

#     new_red_raw = new_ratings_groups[0]
#     new_blue_raw = new_ratings_groups[1]

#     # 3. Calculate Margin of Victory Multiplier (The Adaptation)
#     score_diff = abs(score_red - score_blue)

#     # Example Formula:
#     # Logarithmic scaling allows blowout rewards but prevents infinite scaling.
#     # We add 1 to avoid log(0) or log(1)=0 issues.
#     margin_multiplier = math.log(score_diff + 1, 2)

#     # 4. OT Penalty (Optional)
#     # If it was OT, the "strength" of the win is reduced.
#     if is_overtime:
#         margin_multiplier *= 0.75  # Reduce impact by 25% for OT wins

#     # 5. Apply the Scaled Delta to the Original Ratings
#     # We must manually construct the new rating by scaling the 'jump'

#     final_red = []
#     for old_r, raw_new_r in zip(team_red, new_red_raw):
#         delta_mu = raw_new_r.mu - old_r.mu
#         delta_sigma = raw_new_r.sigma - old_r.sigma

#         # Apply multiplier to the CHANGE (delta), not the absolute value
#         final_mu = old_r.mu + (delta_mu * margin_multiplier)

#         # We usually apply the standard sigma reduction,
#         # but you could scale sigma reduction too if desired.
#         # Here we accept the standard sigma drop as "one match played".
#         final_sigma = raw_new_r.sigma

#         final_red.append(trueskill.Rating(mu=final_mu, sigma=final_sigma))

#     final_blue = []
#     for old_r, raw_new_r in zip(team_blue, new_blue_raw):
#         delta_mu = raw_new_r.mu - old_r.mu
#         # Apply same logic
#         final_mu = old_r.mu + (delta_mu * margin_multiplier)
#         final_sigma = raw_new_r.sigma

#         final_blue.append(trueskill.Rating(mu=final_mu, sigma=final_sigma))

#     return final_red, final_blue


# def process_matches(matches_df):
#     # 1. State Containers
#     # We initialize players "Just In Time" as they appear in the chronological list.
#     elo_ratings = defaultdict(lambda: BASE_ELO)
#     elo_games = defaultdict(int)
#     ts_ratings = defaultdict(lambda: ts_env.create_rating())

#     match_history = []

#     # 2. Chronological Sort (Critical for Time-Series)
#     # matches_df = matches_df.sort_values(by='date').reset_index(drop=True)

#     for idx, row in matches_df.iterrows():
#         # Identify Players
#         p_rd, p_rf = row['red defence'], row['red forward']
#         p_bd, p_bf = row['blue defence'], row['blue forward']

#         # Keys (Distinguishing separate ratings for Defense vs Forward)
#         keys = {
#             'rd': f"{p_rd}_Def", 'rf': f"{p_rf}_Fwd",
#             'bd': f"{p_bd}_Def", 'bf': f"{p_bf}_Fwd"
#         }

#         # --- STEP 3: SNAPSHOT (Capture ratings BEFORE update) ---
#         # This creates the feature row for your ML model
#         match_snapshot = row.to_dict()

#         # Save Elo State
#         match_snapshot['elo_red_def'] = elo_ratings[keys['rd']]
#         match_snapshot['elo_red_fwd'] = elo_ratings[keys['rf']]
#         match_snapshot['elo_blue_def'] = elo_ratings[keys['bd']]
#         match_snapshot['elo_blue_fwd'] = elo_ratings[keys['bf']]

#         # Save TrueSkill State (Mu only is usually sufficient for simple features, but Sigma helps too)
#         match_snapshot['ts_mu_red_def'] = ts_ratings[keys['rd']].mu
#         match_snapshot['ts_mu_red_fwd'] = ts_ratings[keys['rf']].mu
#         match_snapshot['ts_mu_blue_def'] = ts_ratings[keys['bd']].mu
#         match_snapshot['ts_mu_blue_fwd'] = ts_ratings[keys['bf']].mu

#         match_history.append(match_snapshot)

#         # --- Match Context ---
#         result_red = row['result_red']
#         result_blue = row['result_blue']
#         overtime = row['overtime']

#         # Logarithmic margin of victory (diminishing returns for blowout wins)
#         win_diff = max(abs(result_red - result_blue), 1)
#         margin_multiplier = 1 + (math.log(win_diff + 1) / 2)

#         # --- Determine Outcome ---
#         # 0 = Winner, 1 = Loser for TrueSkill ranks
#         if result_red > result_blue:
#             # Red Wins
#             score_red, score_blue = 1, 0
#             ts_ranks = [0, 1]
#         else:
#             # Blue Wins (No draws allowed)
#             score_red, score_blue = 0, 1
#             ts_ranks = [1, 0]

#         # --- ALGORITHM 1: ELO UPDATES ---
#         # Team Averages (Opponent Strength)
#         avg_red_elo = (elo_ratings[keys['rd']] + elo_ratings[keys['rf']]) / 2
#         avg_blue_elo = (elo_ratings[keys['bd']] + elo_ratings[keys['bf']]) / 2

#         # Update Red Team
#         for pos in ['rd', 'rf']:
#             k = get_k_factor(elo_games[keys[pos]], overtime)
#             # Red plays against Blue Average
#             change = calculate_elo_change(
#                 elo_ratings[keys[pos]], avg_blue_elo, score_red, k, margin_multiplier)
#             elo_ratings[keys[pos]] += change
#             elo_games[keys[pos]] += 1

#         # Update Blue Team
#         for pos in ['bd', 'bf']:
#             k = get_k_factor(elo_games[keys[pos]], overtime)
#             # Blue plays against Red Average
#             change = calculate_elo_change(
#                 elo_ratings[keys[pos]], avg_red_elo, score_blue, k, margin_multiplier)
#             elo_ratings[keys[pos]] += change
#             elo_games[keys[pos]] += 1

#         # --- ALGORITHM 2: TRUESKILL UPDATES ---
#         # Group players into teams for calculation
#         red_team = [ts_ratings[keys['rd']], ts_ratings[keys['rf']]]
#         blue_team = [ts_ratings[keys['bd']], ts_ratings[keys['bf']]]

#         # 2. Extract Match Details from the current row
#         # Ensure your DataFrame 'row' has these columns available
#         s_red = row['result_red']
#         s_blue = row['result_blue']
#         is_ot = row['overtime']  # Assuming boolean or recognizable flag

#         # 3. Calculate new ratings using the Custom Function
#         # This replaces: (new_red, new_blue) = ts_env.rate(...)
#         new_red, new_blue = calculate_trueskill_with_margin(
#             team_red=red_team,
#             team_blue=blue_team,
#             score_red=s_red,
#             score_blue=s_blue,
#             is_overtime=is_ot,
#             env=ts_env
#         )

#         # Apply updates back to dictionary
#         ts_ratings[keys['rd']], ts_ratings[keys['rf']] = new_red[0], new_red[1]
#         ts_ratings[keys['bd']], ts_ratings[keys['bf']
#                                            ] = new_blue[0], new_blue[1]

#     history_df = pd.DataFrame(match_history)
#     return history_df



import math
import os
import subprocess
from collections import defaultdict

import pandas as pd
import trueskill

# --- Parameters ---
BASE_ELO = 1000
BASE_K = 32
HIGH_K = 48
LOW_K = 16
EXPERIENCE_THRESHOLD = 100

# --- TrueSkill Setup ---
ts_env = trueskill.TrueSkill(draw_probability=0.0, mu=25.0, sigma=8.333)

def get_k_factor(games_played, overtime):
    base_k = HIGH_K if games_played < EXPERIENCE_THRESHOLD else LOW_K
    return base_k / 2 if overtime else base_k

def calculate_elo_change(r_player, r_opponent_avg, actual_score, k, margin):
    expected = 1 / (1 + 10 ** ((r_opponent_avg - r_player) / 400))
    return k * margin * (actual_score - expected)

def calculate_trueskill_with_margin(team_red, team_blue, score_red, score_blue, is_overtime=False, env=None):
    if env is None: env = trueskill.TrueSkill()
    if score_red > score_blue:
        ranks = [0, 1]
    elif score_blue > score_red:
        ranks = [1, 0]
    else:
        return env.rate([team_red, team_blue], ranks=[0, 0])

    new_ratings_groups = env.rate([team_red, team_blue], ranks=ranks)
    score_diff = abs(score_red - score_blue)
    margin_multiplier = math.log(score_diff + 1, 2)
    if is_overtime: margin_multiplier *= 0.75

    final_red = [trueskill.Rating(mu=old.mu + (raw.mu - old.mu) * margin_multiplier, sigma=raw.sigma) 
                 for old, raw in zip(team_red, new_ratings_groups[0])]
    final_blue = [trueskill.Rating(mu=old.mu + (raw.mu - old.mu) * margin_multiplier, sigma=raw.sigma) 
                  for old, raw in zip(team_blue, new_ratings_groups[1])]
    return final_red, final_blue

def process_matches(matches_df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    elo_games = defaultdict(int)
    ts_ratings = defaultdict(lambda: ts_env.create_rating())
    match_history = []

    for idx, row in matches_df.iterrows():
        p_rd, p_rf = row['red defence'], row['red forward']
        p_bd, p_bf = row['blue defence'], row['blue forward']
        keys = {'rd': f"{p_rd}_Def", 'rf': f"{p_rf}_Fwd", 'bd': f"{p_bd}_Def", 'bf': f"{p_bf}_Fwd"}

        match_snapshot = row.to_dict()
        match_snapshot.update({
            'elo_red_def': elo_ratings[keys['rd']], 'elo_red_fwd': elo_ratings[keys['rf']],
            'elo_blue_def': elo_ratings[keys['bd']], 'elo_blue_fwd': elo_ratings[keys['bf']],
            'ts_mu_red_def': ts_ratings[keys['rd']].mu, 'ts_mu_red_fwd': ts_ratings[keys['rf']].mu,
            'ts_mu_blue_def': ts_ratings[keys['bd']].mu, 'ts_mu_blue_fwd': ts_ratings[keys['bf']].mu
        })
        match_history.append(match_snapshot)

        res_red, res_blue, ot = row['result_red'], row['result_blue'], row['overtime']
        margin_multiplier = 1 + (math.log(max(abs(res_red - res_blue), 1) + 1) / 2)
        s_red, s_blue = (1, 0) if res_red > res_blue else (0, 1)

        avg_red_elo = (elo_ratings[keys['rd']] + elo_ratings[keys['rf']]) / 2
        avg_blue_elo = (elo_ratings[keys['bd']] + elo_ratings[keys['bf']]) / 2

        for pos in ['rd', 'rf']:
            elo_ratings[keys[pos]] += calculate_elo_change(elo_ratings[keys[pos]], avg_blue_elo, s_red, get_k_factor(elo_games[keys[pos]], ot), margin_multiplier)
            elo_games[keys[pos]] += 1
        for pos in ['bd', 'bf']:
            elo_ratings[keys[pos]] += calculate_elo_change(elo_ratings[keys[pos]], avg_red_elo, s_blue, get_k_factor(elo_games[keys[pos]], ot), margin_multiplier)
            elo_games[keys[pos]] += 1

        new_red, new_blue = calculate_trueskill_with_margin([ts_ratings[keys['rd']], ts_ratings[keys['rf']]], [ts_ratings[keys['bd']], ts_ratings[keys['bf']]], res_red, res_blue, ot, ts_env)
        ts_ratings[keys['rd']], ts_ratings[keys['rf']] = new_red
        ts_ratings[keys['bd']], ts_ratings[keys['bf']] = new_blue

    return pd.DataFrame(match_history)



# ... [Keep your BASE_ELO, TrueSkill Setup, and get_k_factor/calculate functions as they were] ...

def run_vitelo_pipeline(df):
    """Bridge to Julia VitELO."""
    
    # We put 'date' first to match how Julia's line 112 expects it
    julia_input = df[['date', 'red defence', 'red forward', 'blue defence', 'blue forward', 
                    'result_red', 'result_blue']].copy()

    # Rename to match the names Julia uses internally
    julia_input.columns = ['date', 'rd', 'rf', 'bd', 'bf', 's_red', 's_blue']

    # Ensure the date format is ISO (standard for CSVs)
    julia_input['date'] = pd.to_datetime(julia_input['date'], dayfirst=True).dt.strftime('%d-%m-%Y')

    print(julia_input)
    

    temp_in = "temp_matches.csv"
    temp_in, temp_out = "temp_matches.csv", "VitELO.csv"
    julia_input.to_csv(temp_in, index=False)

    print("\n--- Starting Julia VitELO Engine ---")
    try:
        subprocess.run(["julia", "./vito/vitelo.jl", temp_in], check=True)
    except Exception as e:
        print(f"Julia execution failed: {e}")
        return None

    if os.path.exists(temp_out):
        results = pd.read_csv(temp_out)
        return results[['player', 'VitELO']]
    return None



if __name__ == "__main__":
    # Load raw data
    match_df = pd.read_excel("./data/results.xlsx", sheet_name="original_data")
    
    # Run Python logic (Elo/TrueSkill)
    time_series_df = process_matches(match_df)
    
    # Run Julia logic (VitELO)
    vitelo_summary = run_vitelo_pipeline(match_df)

    if vitelo_summary is not None:
        # Create a lookup map: { 'PlayerName': Score }
        v_map = vitelo_summary.set_index('player')['VitELO'].to_dict()
        
        # Add VitELO columns to the time_series_df
        # We use the original names from the match row to perform the lookup
        time_series_df['vitelo_red_def'] = time_series_df['red defence'].map(v_map)
        time_series_df['vitelo_red_fwd'] = time_series_df['red forward'].map(v_map)
        time_series_df['vitelo_blue_def'] = time_series_df['blue defence'].map(v_map)
        time_series_df['vitelo_blue_fwd'] = time_series_df['blue forward'].map(v_map)

    # Save the updated sheet
    with pd.ExcelWriter("./data/rankings.xlsx") as writer:
        time_series_df.to_excel(writer, sheet_name='players_skill_time', index=False)
        
    print("Integration complete. Open rankings.xlsx to see all scores.")