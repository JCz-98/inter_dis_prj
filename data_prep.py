import pandas as pd
import numpy as np


def determine_winner(row):
    red_score = row['result_red']
    blue_score = row['result_blue']

    ot_flag = 0
    win_difference = 0

    # determine if it was OT
    if(red_score > 10) or (blue_score > 10):
        ot_flag = 1

    if int(red_score) > int(blue_score):
        winner_label = 'R'
    elif int(blue_score) > int(red_score):
        winner_label = 'B'
    else:
        winner_label = 'D' #handle draws if needed

    win_difference = abs(red_score - blue_score)

    return pd.Series({"winner": winner_label, "overtime": ot_flag, "win_diff": win_difference})

def assign_match_ids(original_df):
    """
    Adds a unique match_id to each row of the original match-level data.
    Format: 'YYYY-MM-DD_m<n>', where <n> is a per-date counter.
    """
    df = original_df.copy()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)  # e.g., "30-01-2024" format

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Generate counter per date
    date_counts = {}
    match_ids = []

    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        if date_str not in date_counts:
            date_counts[date_str] = 1
        else:
            date_counts[date_str] += 1

        match_id = f"{date_str}_m{date_counts[date_str]}"
        match_ids.append(match_id)

    df['match_id'] = match_ids
    return df


def add_extra_columns(csv_file):
    """
    Reads a CSV file, adds a 'winner' label column, and returns a Pandas DataFrame.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with the added 'winner' column.
    """
    try:
        df = pd.read_csv(csv_file)
        # print(df)

        df[["winner", "overtime", "win_diff"]] = df.apply(determine_winner, axis=1)

        # df = assign_match_ids(df)

        return df

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def get_players_statistics(df):
    """
    Analyzes foosball match data and creates a player-centric DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 
                           'red defence', 'red forward', 'blue defence', 
                           'blue forward', 'winner', 'ot'.

    Returns:
        pd.DataFrame: DataFrame with players as rows and performance metrics as columns.
    """

    player_names = pd.unique(df[['red defence', 'red forward', 'blue defence', 'blue forward']].values.ravel('K'))

    player_stats = {}
    for player in player_names:
        player_stats[player] = {
            'blue_played': 0,
            'blue_defense_played': 0,
            'blue_forward_played': 0,
            'blue_won': 0,
            'blue_defense_won': 0,
            'blue_forward_won': 0,
            'blue_lost': 0,
            'blue_defense_lost': 0,
            'blue_forward_lost': 0,
            'red_played': 0,
            'red_defense_played': 0,
            'red_forward_played': 0,
            'red_won': 0,
            'red_defense_won': 0,
            'red_forward_won': 0,
            'red_lost': 0,
            'red_defense_lost': 0,
            'red_forward_lost': 0,
            'overtime': 0,
        }

    for index, row in df.iterrows():
        red_defense = row['red defence']
        red_forward = row['red forward']
        blue_defense = row['blue defence']
        blue_forward = row['blue forward']

        if row['winner'] == 'R':
            winner_team = [red_defense, red_forward]
            loser_team = [blue_defense, blue_forward]
        else:
            winner_team = [blue_defense, blue_forward]
            loser_team = [red_defense, red_forward]

        # Blue Team calculations.
        player_stats[blue_defense]['blue_played'] += 1
        player_stats[blue_forward]['blue_played'] += 1

        player_stats[blue_defense]['blue_defense_played'] += 1
        player_stats[blue_forward]['blue_forward_played'] += 1

        if blue_defense in winner_team:
            player_stats[blue_defense]['blue_won'] += 1
            player_stats[blue_defense]['blue_defense_won'] += 1
        else:
            player_stats[blue_defense]['blue_lost'] += 1
            player_stats[blue_defense]['blue_defense_lost'] += 1

        if blue_forward in winner_team:
            player_stats[blue_forward]['blue_won'] += 1
            player_stats[blue_forward]['blue_forward_won'] += 1
        else:
            player_stats[blue_forward]['blue_lost'] += 1
            player_stats[blue_forward]['blue_forward_lost'] += 1

        # Red Team calculations.
        player_stats[red_defense]['red_played'] += 1
        player_stats[red_forward]['red_played'] += 1

        player_stats[red_defense]['red_defense_played'] += 1
        player_stats[red_forward]['red_forward_played'] += 1

        if red_defense in winner_team:
            player_stats[red_defense]['red_won'] += 1
            player_stats[red_defense]['red_defense_won'] += 1
        else:
            player_stats[red_defense]['red_lost'] += 1
            player_stats[red_defense]['red_defense_lost'] += 1

        if red_forward in winner_team:
            player_stats[red_forward]['red_won'] += 1
            player_stats[red_forward]['red_forward_won'] += 1
        else:
            player_stats[red_forward]['red_lost'] += 1
            player_stats[red_forward]['red_forward_lost'] += 1

        if row['overtime'] == 1:
            for player in [red_defense, red_forward, blue_defense, blue_forward]:
                player_stats[player]['overtime'] += 1

    result_df = pd.DataFrame.from_dict(player_stats, orient='index')
    result_df.index.name = 'player' #setting the index name to player
    result_df.reset_index(inplace = True) #moving the index to a column called player.

    return result_df

import pandas as pd
from collections import defaultdict

def get_team_pair_statistics(df):
    """
    Computes statistics for unique (defender, forward) team pairs in a foosball dataset,
    preserving player roles (defense vs forward).

    Args:
        df (pd.DataFrame): DataFrame with columns 
                           'red defence', 'red forward', 'blue defence', 
                           'blue forward', 'winner', 'overtime'.

    Returns:
        pd.DataFrame: DataFrame with team pairs as rows and performance metrics as columns.
                      Includes 'defender', 'forward', 'played', 'won', 'lost', 'overtime', 'side'.
    """

    team_stats = defaultdict(lambda: {
        'played': 0,
        'won': 0,
        'lost': 0,
        'overtime': 0,
        'side': ''
    })

    for _, row in df.iterrows():
        red_team = (row['red defence'], row['red forward'])   # roles preserved
        blue_team = (row['blue defence'], row['blue forward'])

        winner = row['winner']
        ot = row['overtime']

        # Update red team
        team_stats[red_team]['played'] += 1
        team_stats[red_team]['side'] = 'red'
        if winner == 'R':
            team_stats[red_team]['won'] += 1
        else:
            team_stats[red_team]['lost'] += 1
        if ot == 1:
            team_stats[red_team]['overtime'] += 1

        # Update blue team
        team_stats[blue_team]['played'] += 1
        team_stats[blue_team]['side'] = 'blue'
        if winner == 'B':
            team_stats[blue_team]['won'] += 1
        else:
            team_stats[blue_team]['lost'] += 1
        if ot == 1:
            team_stats[blue_team]['overtime'] += 1

    # Build result DataFrame
    result_df = pd.DataFrame([
        {
            'defender': team[0],
            'forward': team[1],
            **stats
        }
        for team, stats in team_stats.items()
    ])

    return result_df


def calculate_win_loss_percentages(player_stats_df):
    """
    Calculates raw and weighted win/loss percentages based on player statistics.
    No scaling is applied; only weighting based on total games played.
    """
    percentages_df = pd.DataFrame()
    percentages_df['player'] = player_stats_df['player']

    # Total and max games played per player
    total_games = player_stats_df['blue_played'] + player_stats_df['red_played']
    max_games = total_games.max()
    weights = total_games / max_games

    def safe_divide(numerator, denominator):
        return numerator / denominator.replace(0, np.nan)

    # Raw percentages
    raw = {
        'win_loss_percentage': safe_divide(
            player_stats_df['blue_won'] + player_stats_df['red_won'],
            player_stats_df['blue_lost'] + player_stats_df['red_lost']
        ),
        'win_percentage': safe_divide(
            player_stats_df['blue_won'] + player_stats_df['red_won'],
            total_games
        ),
        'win_blue_percentage': safe_divide(player_stats_df['blue_won'], player_stats_df['blue_played']),
        'win_defense_blue_percentage': safe_divide(player_stats_df['blue_defense_won'], player_stats_df['blue_won']),
        'win_forward_blue_percentage': safe_divide(player_stats_df['blue_forward_won'], player_stats_df['blue_won']),
        'loss_percentage': safe_divide(
            player_stats_df['blue_lost'] + player_stats_df['red_lost'],
            total_games
        ),
        'loss_blue_percentage': safe_divide(player_stats_df['blue_lost'], player_stats_df['blue_played']),
        'loss_defense_blue_percentage': safe_divide(player_stats_df['blue_defense_lost'], player_stats_df['blue_lost']),
        'loss_forward_blue_percentage': safe_divide(player_stats_df['blue_forward_lost'], player_stats_df['blue_lost']),
        'win_red_percentage': safe_divide(player_stats_df['red_won'], player_stats_df['red_played']),
        'win_defense_red_percentage': safe_divide(player_stats_df['red_defense_won'], player_stats_df['red_won']),
        'win_forward_red_percentage': safe_divide(player_stats_df['red_forward_won'], player_stats_df['red_won']),
        'loss_red_percentage': safe_divide(player_stats_df['red_lost'], player_stats_df['red_played']),
        'loss_defense_red_percentage': safe_divide(player_stats_df['red_defense_lost'], player_stats_df['red_lost']),
        'loss_forward_red_percentage': safe_divide(player_stats_df['red_forward_lost'], player_stats_df['red_lost']),
    }

    # Apply weighting but no scaling
    for col, series in raw.items():
        percentages_df[col] = (series * weights).fillna(0)

    return percentages_df



# Example usage:
csv_file_path = 'data/scores.csv'  # Replace with your CSV file path.
result_df = add_extra_columns(csv_file_path)
print(result_df)
player_statistics = get_players_statistics(result_df)
team_statistics = get_team_pair_statistics(result_df)
player_percentages = calculate_win_loss_percentages(player_statistics)


# print(player_statistics)
# print(team_statistics)
# print(player_percentages)

# print(player_statistics.columns)
# print(team_statistics.columns)
# print(player_percentages.columns)




with pd.ExcelWriter("./data/results.xlsx") as writer:
    result_df.to_excel(writer, sheet_name='original_data', index=False)
    player_statistics.to_excel(writer, sheet_name='player_statistics', index=False)
    team_statistics.to_excel(writer, sheet_name='team_statistics', index=False)
    player_percentages.to_excel(writer, sheet_name='player_percentages', index=False)



# print_result_by_teamcolor(player_percentages, 'blue')
# print_result_by_teamcolor(player_percentages, 'red')



    # Optionally, save the modified DataFrame back to a CSV file:
    # result_df.to_csv('your_data_with_winner.csv', index=False)