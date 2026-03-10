import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

def extract_player_series(df, player, metric_prefix='elo'):
    """Helper to melt the wide match dataframe into a long player time-series."""
    # Maps the column names in the history_df to a standard format
    positions = [
        {'role': 'Def', 'side': 'Red', 'p_col': 'red defence', 'r_col': f'{metric_prefix}_red_def'},
        {'role': 'Fwd', 'side': 'Red', 'p_col': 'red forward', 'r_col': f'{metric_prefix}_red_fwd'},
        {'role': 'Def', 'side': 'Blue', 'p_col': 'blue defence', 'r_col': f'{metric_prefix}_blue_def'},
        {'role': 'Fwd', 'side': 'Blue', 'p_col': 'blue forward', 'r_col': f'{metric_prefix}_blue_fwd'},
    ]
    
    records = []
    for _, row in df.iterrows():
        for pos in positions:
            if row[pos['p_col']] == player:
                records.append({
                    'date': row['date'],
                    'rating': row[pos['r_col']],
                    'role': pos['role'],
                    'side': pos['side']
                })
    return pd.DataFrame(records)

def plot_player_history(history_df, player_name, metric='elo', aggregated=False):
    """
    Plots the ranking development for a specific player.
    
    Args:
        history_df: dataframe
        metric (str): 'elo', 'ts_mu', or 'vitelo'
        aggregated (bool): If True, averages Def/Fwd ratings into one line.
    """
    data = extract_player_series(history_df, player_name, metric)
    
    if data.empty:
        print(f"No data found for player: {player_name}")
        return

    # Ensure date is datetime object and sort
    data['date'] = pd.to_datetime(data['date'], dayfirst=True)
    data = data.sort_values('date')

    # If aggregated, we stick to the single plot logic (simplified for this context)
    if aggregated:
        plt.figure(figsize=(12, 6))
        agg_data = data.groupby('date')['rating'].mean().reset_index()
        # Use simple Month Formatting for aggregated too
        ax = plt.gca()
        ax.plot(agg_data['date'], agg_data['rating'], marker='o', label=f'{player_name} (Aggregated)')
        
        # Date Formatting
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))
        
        plt.title(f"{metric.upper()} Rating History: {player_name} (Aggregated)")
        plt.xlabel("Date")
        plt.ylabel("Rating")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # ---------------------------------------------------------
    # NON-AGGREGATED: STACKED RED/BLUE GRAPHS
    # ---------------------------------------------------------
    
    # Create 2 vertically stacked subplots, sharing the X-axis (Time)
    fig, (ax_red, ax_blue) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Define Marker Mapping: Triangle (^) for Fwd, Square (s) for Def
    # Handling potential case sensitivity (Def/def)
    markers = {'Fwd': '^', 'fwd': '^', 'Def': 'o', 'def': 'o'}

    # --- TOP GRAPH: RED TEAM ---
    red_data = data[data['side'].str.lower() == 'red']
    red_colors = {"Fwd": '#d62728', "Def": "#961b1b"}
    # Iterate through roles found in the Red data
    for role in red_data['role'].unique():
        subset = red_data[red_data['role'] == role]
        marker = markers.get(role, 'x') # Default to circle if unknown
        
        ax_red.plot(
            subset['date'], 
            subset['rating'], 
            color= red_colors[role],     # Red Color
            marker=marker,       # Square or Triangle
            linestyle='-', 
            linewidth=1.5,
            markersize=3,
            label=f"Red {role}"
        )

    ax_red.set_title(f"{player_name} - RED SIDE (Top)")
    ax_red.set_ylabel(f"{metric.upper()} Rating")
    ax_red.grid(True, alpha=0.3, linestyle='--')
    ax_red.legend(loc='upper left')

    # --- BOTTOM GRAPH: BLUE TEAM ---
    blue_data = data[data['side'].str.lower() == 'blue']
    blue_colors = {"Fwd": '#1f77b4', "Def": "#56A0D3"}
    
    # Iterate through roles found in the Blue data
    for role in blue_data['role'].unique():
        subset = blue_data[blue_data['role'] == role]
        marker = markers.get(role, 'o')
        
        ax_blue.plot(
            subset['date'], 
            subset['rating'], 
            color=blue_colors[role],     # Blue Color
            marker=marker,       # Square or Triangle
            linestyle='-', 
            linewidth=1.5,
            markersize=3,
            label=f"Blue {role}"
        )

    ax_blue.set_title(f"{player_name} - BLUE SIDE (Bottom)")
    ax_blue.set_xlabel("Date")
    ax_blue.set_ylabel(f"{metric.upper()} Rating")
    ax_blue.grid(True, alpha=0.3, linestyle='--')
    ax_blue.legend(loc='upper left')

    # --- X-AXIS FORMATTING (MONTHLY INTERVALS) ---
    # Apply to the bottom axis (ax_blue), but since sharex=True, it affects alignment
    locator = mdates.MonthLocator(interval=1)  # Tick every 1 month
    formatter = mdates.DateFormatter('%m.%y')  # Format: Month.Year (e.g., 03.24)

    ax_blue.xaxis.set_major_locator(locator)
    ax_blue.xaxis.set_major_formatter(formatter)

    # Rotate dates slightly for readability
    plt.setp(ax_blue.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def plot_team_history(history_df, player_names, metric='elo'):
    """
    Plots the aggregated rating for a specific pair of players.
    
    Args:
        player_names (list): ['PlayerA', 'PlayerB']
    """
    p1, p2 = player_names
    records = []
    
    # Map rating columns
    r_cols = {
        'rd': f'{metric}_red_def', 'rf': f'{metric}_red_fwd',
        'bd': f'{metric}_blue_def', 'bf': f'{metric}_blue_fwd'
    }

    for _, row in history_df.iterrows():
        # Check if pair is on Red Team (regardless of who is Def/Fwd)
        is_red = (row['red defence'] in player_names) and (row['red forward'] in player_names)
        # Check if pair is on Blue Team
        is_blue = (row['blue defence'] in player_names) and (row['blue forward'] in player_names)
        
        rating_val = None
        
        if is_red:
            rating_val = (row[r_cols['rd']] + row[r_cols['rf']]) / 2
        elif is_blue:
            rating_val = (row[r_cols['bd']] + row[r_cols['bf']]) / 2
            
        if rating_val is not None:
            records.append({'date': row['date'], 'rating': rating_val})

    if not records:
        print(f"No matches found for team: {player_names}")
        return

    team_df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=team_df, x='date', y='rating', marker='o', color='purple')
    plt.title(f"{metric.upper()} Team Rating: {' & '.join(player_names)}")
    plt.xlabel("Date")
    plt.ylabel("Combined Rating")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    time_df = pd.read_excel("./data/rankings.xlsx", sheet_name="players_skill_time")

    plot_player_history(time_df, "Leonardo", "elo", aggregated=False)