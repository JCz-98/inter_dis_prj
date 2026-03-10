import pandas as pd
import numpy as np
from math import log10

# Initialize ratings per player/side/role
def initialize_ratings(player_stats, player_percentages):
    base_rating = 1000
    ratings = {}

    for _, row in player_stats.iterrows():
        player = row['player']
        
        # Use win percentage as a bias to initial rating
        pct = player_percentages[player_percentages['player'] == player]
        wp = pct['win_percentage'].values[0] if not pct.empty else 0.5
        
        rating_bias = (wp - 0.5) * 400  # ±200 swing
        for side in ['blue', 'red']:
            for role in ['defense', 'forward']:
                key = f"{player}_{side}_{role}"
                ratings[key] = base_rating + rating_bias
    return ratings

# Elo update function
def update_rating(r1, r2, score, k=32):
    expected_r1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    new_r1 = r1 + k * (score - expected_r1)
    return new_r1

# Main Elo calculation from team stats
def compute_elo(team_stats, player_stats, player_percentages):
    ratings = initialize_ratings(player_stats, player_percentages)

    for _, row in team_stats.iterrows():
        d = row['defender']
        f = row['forward']
        side = row['side'].lower()
        team_rating = (ratings.get(f"{d}_{side}_defense", 1000) + 
                       ratings.get(f"{f}_{side}_forward", 1000)) / 2

        # Opponent side and keys
        opp_side = 'red' if side == 'blue' else 'blue'
        # Simulate opponent team rating as average for simplicity
        opp_rating = 1000

        # Match result: 1 = win, 0 = loss
        result = 1 if row['won'] > row['lost'] else 0

        # Adjust K-factor if overtime
        k = 16 if row['overtime'] else 32

        # Update both players
        for player, role in [(d, 'defense'), (f, 'forward')]:
            key = f"{player}_{side}_{role}"
            current_rating = ratings.get(key, 1000)
            ratings[key] = update_rating(current_rating, opp_rating, result, k)

    return ratings


if __name__ == "__main__":

    # Path to your Excel file
    excel_file = './results.xlsx'  # Replace with actual path

    # Load each sheet into its corresponding DataFrame
    player_statistics = pd.read_excel(excel_file, sheet_name='player_statistics')
    team_statistics = pd.read_excel(excel_file, sheet_name='team_statistics')
    player_percentages = pd.read_excel(excel_file, sheet_name='player_percentages')

    print(player_statistics)
    print(team_statistics)
    print(player_percentages)

    # Compute Elo ratings
    ratings = compute_elo(team_statistics, player_statistics, player_percentages)
    print("Updated Ratings:")
    for player, rating in ratings.items():
        print(f"{player}: {rating:.2f}")

        