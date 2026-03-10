import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict

# --- Updated Elo Parameters ---
BASE_RATING = 1000
BASE_K = 32
HIGH_K = 48  # for new players
LOW_K = 16    # for experienced players
EXPERIENCE_THRESHOLD = 100  # games to be considered experienced

# --- Initialize Ratings ---
def initialize_ratings(player_stats_df):
    ratings = {}
    games_played = {}

    for _, row in player_stats_df.iterrows():
        player = row['player']
        total_games = row['red_played'] + row['blue_played']
        for side in ['red', 'blue']:
            for role in ['defense', 'forward']:
                key = f"{player}_{side}_{role}"
                ratings[key] = BASE_RATING
                games_played[key] = total_games
    # print(ratings)
    print(games_played)
    return ratings, games_played

# --- Calculate Dynamic K-factor ---
def get_k_factor(key, games_played, overtime):
    base_k = HIGH_K if games_played.get(key, 0) < EXPERIENCE_THRESHOLD else LOW_K
    if overtime:
        return base_k / 2  # reduce K if match is very close
    return base_k

# --- Elo Update ---
def update_rating(r1, r2, score, k, margin=1):
    expected_r1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    new_r1 = r1 + k * margin * (score - expected_r1)  # weighted update  <-- updated here
    return new_r1

# --- Compute Elo ---
def compute_elo(original_df, player_stats_df):
    ratings, games_played = initialize_ratings(player_stats_df)
    team_ratings = defaultdict(lambda: BASE_RATING)

    for _, row in original_df.iterrows():
        # Players
        rd, rf = row['red defence'], row['red forward']
        bd, bf = row['blue defence'], row['blue forward']

        # Keys for each player
        keys = {
            'red_def': f"{rd}_red_defense",
            'red_for': f"{rf}_red_forward",
            'blue_def': f"{bd}_blue_defense",
            'blue_for': f"{bf}_blue_forward"
        }

        # Average team ratings
        red_team_rating = (ratings[keys['red_def']] + ratings[keys['red_for']]) / 2
        blue_team_rating = (ratings[keys['blue_def']] + ratings[keys['blue_for']]) / 2

        # Score & Result
        result_red = row['result_red']
        result_blue = row['result_blue']
        winner = row['winner']
        overtime = row['overtime']
        win_diff = max(abs(result_red - result_blue), 1)  # avoid 0
        margin = 1 + (win_diff / 10)  # weight impact by goal diff  <-- updated here

        # Determine result
        red_result = 1 if winner == 'R' else 0
        blue_result = 1 - red_result

        # Update individual player ratings
        for key in ['red_def', 'red_for']:
            k_val = get_k_factor(keys[key], games_played, overtime)
            ratings[keys[key]] = update_rating(
                ratings[keys[key]], blue_team_rating, red_result, k_val, margin
            )

        for key in ['blue_def', 'blue_for']:
            k_val = get_k_factor(keys[key], games_played, overtime)
            ratings[keys[key]] = update_rating(
                ratings[keys[key]], red_team_rating, blue_result, k_val, margin
            )

        # --- Update team ratings ---  <-- new logic for team Elo
        red_team_key = f"{rd}+{rf}_red"
        blue_team_key = f"{bd}+{bf}_blue"

        team_k = BASE_K / 2 if overtime else BASE_K

        team_ratings[red_team_key] = update_rating(
            team_ratings[red_team_key], team_ratings[blue_team_key], red_result, team_k, margin
        )
        team_ratings[blue_team_key] = update_rating(
            team_ratings[blue_team_key], team_ratings[red_team_key], blue_result, team_k, margin
        )

    return ratings, team_ratings



def get_individual_rankings(ratings):
    """
    Processes individual player ratings into a tidy DataFrame with ranked positions
    by side and role.

    Parameters:
        ratings (dict): Dictionary of individual ratings with keys in the form 'player_side_role'.

    Returns:
        pd.DataFrame: DataFrame with structure [name, red_defense_rank, red_forward_rank, blue_defense_rank, blue_forward_rank]
    """
    # Parse ratings into DataFrame
    data = []
    for key, rating in ratings.items():
        try:
            player, side, role = key.split('_')
            data.append({'player': player, 'side': side, 'role': role, 'rating': round(rating)})
        except ValueError:
            continue  # skip malformed keys
    
    df = pd.DataFrame(data)

    # Pivot into wide format
    pivot = df.pivot_table(index='player', columns=['side', 'role'], values='rating')

    # Rename columns for clarity
    pivot.columns = [f"{side}_{role}_rank" for side, role in pivot.columns]

    # Compute average rank across available roles (optional)
    expected_cols = ['red_defense_rank', 'red_forward_rank', 'blue_defense_rank', 'blue_forward_rank']
    existing_cols = [col for col in expected_cols if col in pivot.columns]
    pivot['average'] = pivot[existing_cols].mean(axis=1)

    # Sort by average rank (descending = better)
    pivot = pivot.sort_values(by='average', ascending=False).reset_index()

    return pivot



def get_team_rankings(team_ratings):
    """
    Processes team ratings dictionary into a sorted DataFrame.

    Parameters:
        team_ratings (dict): Dictionary with keys like 'PlayerA+PlayerB_red' and rating as value.

    Returns:
        pd.DataFrame: DataFrame with columns [team, color, ranking], sorted by ranking descending.
    """
    records = []
    for key, rating in team_ratings.items():
        try:
            team_part, color = key.rsplit('_', 1)
            records.append({'team': team_part, 'color': color, 'ranking': round(rating)})
        except ValueError:
            continue  # skip malformed keys

    df = pd.DataFrame(records)
    df = df.sort_values(by='ranking', ascending=False).reset_index(drop=True)
    return df



# Example usage
if __name__ == "__main__":

    match_df = pd.read_excel("./data/results.xlsx", sheet_name="original_data")
    player_stats_df = pd.read_excel("./data/results.xlsx", sheet_name="player_statistics")
    individual_ratings, team_ratings = compute_elo(match_df, player_stats_df)

    # print("Individual Ratings:")
    # for player_key, rating in individual_ratings.items():
    #     print(f"{player_key}: {rating:.2f}")

    # print("\nTeam Ratings:")
    # for team_key, rating in team_ratings.items():
    #     print(f"{team_key}: {rating:.2f}")

    individual_df = get_individual_rankings(individual_ratings)
    team_df = get_team_rankings(team_ratings)

    print(individual_df)
    print(team_df)

    # Save the results to Excel
    with pd.ExcelWriter("./data/rankings.xlsx") as writer:
        individual_df.to_excel(writer, sheet_name='individual_rankings', index=False)
        team_df.to_excel(writer, sheet_name='team_rankings', index=False)

    

