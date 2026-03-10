import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom, poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# def transform_foosball_data(historic_df, metric='elo', split_ratio=0.8, test_month=11):
#     """
#     Loads foosball match data, sorts by date, splits into train/test sets,
#     and then stacks them separately for GLM training.
    
#     Args:
#         file_path (str): Path to the rankings file.
#         metric (str): The prefix of the rating columns to use ('elo' or 'ts_mu').
#         split_ratio (float): Fraction of oldest matches to use for training.
                      
#     Returns:
#         tuple: (stacked_train_df, stacked_test_df)
#     """
#     try:
#         # Load the dataset
#         # Using read_excel as per your snippet
#         df = historic_df
        
#         # 0. Pre-Process: Calculate Sums (Row-wise operation, safe before split)
#         r_def_col = f'{metric}_red_def'
#         r_fwd_col = f'{metric}_red_fwd'
#         b_def_col = f'{metric}_blue_def'
#         b_fwd_col = f'{metric}_blue_fwd'
        
#         df['TeamRed_Sum'] = df[r_def_col] + df[r_fwd_col]
#         df['TeamBlue_Sum'] = df[b_def_col] + df[b_fwd_col]

#         # 1. Sort by Date to ensure Time-Series integrity
#         # # (Assuming a 'date' column exists; if not, relies on file order)
#         # if 'date' in df.columns:
#         #     df['date'] = pd.to_datetime(df['date'])
#         #     df = df.sort_values(by='date').reset_index(drop=True)
            
#         # 2. Split the "Matches" DataFrame (Time-Series Split)
#         # We split the raw matches so complete games stay together
#         # split_index = int(len(df) * split_ratio)
        
#         # train_matches = df.iloc[:split_index].copy()
#         # test_matches = df.iloc[split_index:].copy()

#         # 2. Split Logic (Modified)
#         if test_month is not None:
#             # --- MONTHLY CROSS-VALIDATION SPLIT ---
#             # Train: All history strictly BEFORE the test month
#             train_mask = pd.to_datetime(df['date'], dayfirst=True).dt.month < test_month
#             # Test: Only the specific test month
#             test_mask = pd.to_datetime(df['date'], dayfirst=True).dt.month == test_month
            
#             train_matches = df[train_mask].copy()
#             test_matches = df[test_mask].copy()

#             print(train_matches)
#             print(test_matches)
            
#             print(f"CV Split (Month {test_month}): {len(train_matches)} Train / {len(test_matches)} Test matches")
            
#         else:
#             # --- STANDARD RATIO SPLIT ---
#             split_index = int(len(df) * split_ratio)
#             train_matches = df.iloc[:split_index].copy()
#             test_matches = df.iloc[split_index:].copy()
#             print(f"Ratio Split ({split_ratio}): {len(train_matches)} Train / {len(test_matches)} Test matches")
        
#         print(f"Time-Series Split: {len(train_matches)} Oldest Matches (Train) / {len(test_matches)} Newest Matches (Test)")

#         # 3. Define Stacking Logic (Helper)
#         def stack_matches(match_df):
#             # Red Perspective
#             red_df = pd.DataFrame()
#             red_df['goals'] = match_df['result_red']
#             red_df['skill_diff'] = match_df['TeamRed_Sum'] - match_df['TeamBlue_Sum'] 
#             red_df['team'] = 'Red'
            
#             # Blue Perspective
#             blue_df = pd.DataFrame()
#             blue_df['goals'] = match_df['result_blue']
#             blue_df['skill_diff'] = match_df['TeamBlue_Sum'] - match_df['TeamRed_Sum']
#             blue_df['team'] = 'Blue'
            
#             # Concatenate
#             return pd.concat([red_df, blue_df], axis=0).reset_index(drop=True)

#         # 4. Stack the datasets separately
#         stacked_train_df = stack_matches(train_matches)
#         stacked_test_df = stack_matches(test_matches)
        
#         return stacked_train_df, stacked_test_df

#     except FileNotFoundError:
#         print(f"Error: The file at {file_path} was not found.")
#         return None, None
#     except KeyError as e:
#         print(f"Error: Missing column in dataset - {e}")
#         return None, None


def transform_foosball_data(historic_df, metric='elo', split_ratio=0.8, test_period=None):
    """
    Loads foosball match data, sorts by date, splits into train/test sets,
    and then stacks them separately for GLM training.
    
    Args:
        historic_df (pd.DataFrame): The rankings dataframe.
        metric (str): The prefix of the rating columns to use ('elo' or 'ts_mu').
        split_ratio (float): Fraction of oldest matches to use for training (fallback).
        test_period (str): 'YYYY-MM' string to define the specific test month.
                      
    Returns:
        tuple: (stacked_train_df, stacked_test_df)
    """
    try:
        df = historic_df.copy()
        
        # 0. Pre-Process: Calculate Sums
        r_def_col = f'{metric}_red_def'
        r_fwd_col = f'{metric}_red_fwd'
        b_def_col = f'{metric}_blue_def'
        b_fwd_col = f'{metric}_blue_fwd'
        
        df['TeamRed_Sum'] = df[r_def_col] + df[r_fwd_col]
        df['TeamBlue_Sum'] = df[b_def_col] + df[b_fwd_col]

        # 1. Sort by Date to ensure Time-Series integrity
        # if 'date' in df.columns:
        #     # Parse dates strictly and sort
        #     df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        #     df = df.sort_values(by='date').reset_index(drop=True)
            
        # 2. Split Logic (Modified for Year-Month rolling windows)
        if test_period is not None:
            # --- MONTHLY CROSS-VALIDATION SPLIT ---
            # Convert the target string (e.g. '2023-11') to a monthly period
            target_period = pd.to_datetime(test_period).to_period('M')
            
            # Convert dataframe dates to monthly periods
            match_periods = pd.to_datetime(df['date'], dayfirst=True).dt.to_period('M')
            
            # Train: All history strictly BEFORE the test month/year
            train_mask = match_periods < target_period
            # Test: Only the specific test month/year
            test_mask = match_periods == target_period
            
            train_matches = df[train_mask].copy()
            test_matches = df[test_mask].copy()
            
            print(f"CV Split (Period {test_period}): {len(train_matches)} Train / {len(test_matches)} Test matches")
            
        else:
            # --- STANDARD RATIO SPLIT ---
            split_index = int(len(df) * split_ratio)
            train_matches = df.iloc[:split_index].copy()
            test_matches = df.iloc[split_index:].copy()
            print(f"Ratio Split ({split_ratio}): {len(train_matches)} Train / {len(test_matches)} Test matches")

        # 3. Define Stacking Logic (Helper)
        def stack_matches(match_df):
            # Red Perspective
            red_df = pd.DataFrame()
            red_df['goals'] = match_df['result_red']
            red_df['skill_diff'] = match_df['TeamRed_Sum'] - match_df['TeamBlue_Sum'] 
            red_df['team'] = 'Red'
            
            # Blue Perspective
            blue_df = pd.DataFrame()
            blue_df['goals'] = match_df['result_blue']
            blue_df['skill_diff'] = match_df['TeamBlue_Sum'] - match_df['TeamRed_Sum']
            blue_df['team'] = 'Blue'
            
            # Concatenate
            return pd.concat([red_df, blue_df], axis=0).reset_index(drop=True)

        # 4. Stack the datasets separately
        stacked_train_df = stack_matches(train_matches)
        stacked_test_df = stack_matches(test_matches)
        
        return stacked_train_df, stacked_test_df

    except KeyError as e:
        print(f"Error: Missing column in dataset - {e}")
        return None, None



def split_and_normalize(train_df, test_df):
    """
    Extracts features/targets from pre-split dataframes and normalizes 
    using only training statistics.
    
    Args:
        train_df (pd.DataFrame): Stacked training data.
        test_df (pd.DataFrame): Stacked testing data.
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # 1. Extract Features (X) and Target (y)
    # We use .values.reshape(-1, 1) to ensure X is a 2D array for the Scaler
    X_train_raw = train_df['skill_diff'].values.reshape(-1,1)
    y_train = train_df['goals'].values
    
    X_test_raw = test_df['skill_diff'].values.reshape(-1,1)
    y_test = test_df['goals'].values
    
    # 2. FIT the scaler on TRAINING data only
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    
    # 3. TRANSFORM both sets
    # We apply the mean/std learned from Train to the Test set
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Data Prepared: {len(X_train)} training observations, {len(X_test)} test observations.")
    print(f"Scaler Mean: {scaler.mean_[0]:.4f}, Scale (Std): {scaler.scale_[0]:.4f}")
    
    return X_train, X_test, y_train, y_test, scaler


def train_glms(X_train, y_train):
    """
    Trains Poisson and Negative Binomial models using statsmodels.
    
    Args:
        X_train (np.array): Normalized feature matrix.
        y_train (np.array): Target vector (goals).
        
    Returns:
        models (dict): Dictionary containing the fitted model objects.
        metrics (pd.DataFrame): Comparison of AIC, BIC, and Converge status.
    """
    # 1. Add Intercept (Constant)
    # This creates a column of 1s so the model can learn a baseline goal rate.
    X_train_const = sm.add_constant(X_train)
    
    models = {}
    results_list = []
    
    print("--- Training Models ---")

    # --- Model A: Poisson Regression ---
    try:
        print("Fitting Poisson Model...\n")
        # GLM with Poisson family and Log link function
        poisson_model = sm.GLM(y_train, X_train_const, family=sm.families.Poisson())
        poisson_results = poisson_model.fit()

        print(poisson_results.summary())
        
        models['Poisson'] = poisson_results
        print("Done.")
        
        results_list.append({
            'Model': 'Poisson',
            'Converged': poisson_results.converged,
            'AIC': poisson_results.aic,
            'BIC': poisson_results.bic_llf,
            'Log-Likelihood': poisson_results.llf
        })
        
    except Exception as e:
        print(f"\nPoisson Training Failed: {e}")

    # --- Model B: Negative Binomial Regression ---
    try:
        print("Fitting Negative Binomial Model...\n")
        # We use the discrete model class to estimate the dispersion parameter (alpha)
        # loglike_method='nb2' is the standard parameterization
        nb_model = sm.NegativeBinomial(y_train, X_train_const, loglike_method='nb2')
        nb_results = nb_model.fit(disp=0) # disp=0 turns off internal convergence print logs
        
        print(nb_results.summary())

        models['NegativeBinomial'] = nb_results
        print("Done.")
        
        results_list.append({
            'Model': 'Negative Binomial',
            'Converged': nb_results.mle_retvals['converged'], # metrics stored differently in discrete model
            'AIC': nb_results.aic,
            'BIC': nb_results.bic,
            'Log-Likelihood': nb_results.llf
        })

    except Exception as e:
        print(f"\nNegative Binomial Training Failed: {e}")

    # --- Summary ---
    metrics_df = pd.DataFrame(results_list)
    
    print("\n--- Training Evaluation ---")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
        
        # Heuristic for checking which model is better
        best_model = metrics_df.loc[metrics_df['AIC'].idxmin()]
        print(f"\nRecommendation: The {best_model['Model']} model has the lower AIC " 
              f"({best_model['AIC']:.2f}) and fits the data better.")
    else:
        print("No models converged.")

    return models, metrics_df

# def evaluate_on_test_data(model, X_test, y_test):
#     """
#     Generates predictions for the test set and calculates error metrics.
    
#     Args:
#         model (statsmodels result): The trained GLM object.
#         X_test (np.array): Normalized features for the test set.
#         y_test (np.array): Actual goals scored in the test set.
        
#     Returns:
#         pd.DataFrame: A summary of the evaluation metrics.
#     """
#     # 1. Prepare Input (Add Constant)
#     # Statsmodels requires the intercept column explicitly
#     X_test_const = sm.add_constant(X_test)
    
#     # 2. Predict
#     # Returns the expected goals (Lambda) for each match in the test set
#     y_pred_expected = model.predict(X_test_const)
    
#     # 3. Calculate Metrics
#     mae = mean_absolute_error(y_test, y_pred_expected)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred_expected))
    
#     # 4. Calculate Probabilistic Metrics (Log Loss & Perplexity)
#     # For every test observation, calculate the probability of the ACTUAL outcome
#     # given the predicted lambda.
#     # P(k=actual | lambda=predicted)
#     prob_of_actual = poisson.pmf(y_test, y_pred_expected)
    
#     # Clip probability to avoid log(0) error (epsilon)
#     epsilon = 1e-15
#     prob_of_actual = np.maximum(prob_of_actual, epsilon)
    
#     # Log Loss (Negative Log Likelihood per sample)
#     log_loss_vals = -np.log(prob_of_actual)
#     avg_log_loss = np.mean(log_loss_vals)
    
#     # Perplexity = exp(Log Loss)
#     # Intuitively: How "confused" is the model? 
#     # If Perplexity is 5, the model is as unsure as rolling a 5-sided die.
#     perplexity = np.exp(avg_log_loss)
    
#     # 5. Calibration Check
#     avg_actual = np.mean(y_test)
#     avg_predicted = np.mean(y_pred_expected)
#     bias = avg_predicted - avg_actual
    
#     # 6. Construct Output Report
#     metrics = {
#         'Metric': [
#             'Mean Absolute Error (MAE)', 
#             'Root Mean Squared Error (RMSE)', 
#             'Log Loss (NLL)', 
#             'Perplexity',
#             'Avg Goals (Actual)', 
#             'Avg Goals (Predicted)', 
#             'Prediction Bias'
#         ],
#         'Value': [mae, rmse, avg_log_loss, perplexity, avg_actual, avg_predicted, bias]
#     }
    
#     # print(f"--- Test Set Evaluation: {model_name} ---")
#     results_df = pd.DataFrame(metrics)
#     # print(results_df.to_string(index=False, float_format="%.4f"))

#     return results_df


# def evaluate_on_test_data(model, X_test, y_test):
#     """
#     Generates predictions for the test set and calculates error metrics.
    
#     Args:
#         model (statsmodels result): The trained GLM object.
#         X_test (np.array/pd.DataFrame): Features for the test set.
#         y_test (np.array/pd.Series): Actual goals scored in the test set.
        
#     Returns:
#         pd.DataFrame: A summary of the evaluation metrics.
#     """
#     # 1. Prepare Input (Add Constant)
#     # Statsmodels requires the intercept column explicitly
#     X_test_const = sm.add_constant(X_test, has_constant='add')
    
#     # 2. Predict Expected Goals (Lambda / Mu)
#     y_pred_expected = model.predict(X_test_const)
    
#     # 3. Calculate Error Metrics
#     mae = mean_absolute_error(y_test, y_pred_expected)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred_expected))
    
#     # 4. Calculate Probabilistic Metrics (Log Loss & Perplexity)
#     # Dynamically check if the model is Negative Binomial or Poisson
#     # model_family_str = str(getattr(getattr(model, 'model', None), 'family', type(getattr(model, 'model', None))))
    
#     # Check if the model is the statsmodels Discrete NegativeBinomial
#     if type(model.model).__name__ == 'NegativeBinomial':
#         # Extract the learned dispersion parameter (alpha) directly from the fitted parameters
#         alpha = alpha = model.params[-1]
            
#         # Convert statsmodels mean (mu) and dispersion (alpha) to scipy nbinom parameters (n, p)
#         # Statsmodels variance: Var = mu + alpha * mu^2
#         # Scipy parameterization: n = number of successes, p = probability of success
#         n_param = 1.0 / alpha
#         p_param = 1.0 / (1.0 + alpha * y_pred_expected)
        
#         prob_of_actual = nbinom.pmf(y_test, n_param, p_param)
        
#     else:
#         # Default to Poisson assumption
#         prob_of_actual = poisson.pmf(y_test, y_pred_expected)
    
#     # Clip probability to avoid log(0) error (epsilon)
#     epsilon = 1e-15
#     prob_of_actual = np.maximum(prob_of_actual, epsilon)
    
#     # Log Loss (Negative Log Likelihood per sample)
#     log_loss_vals = -np.log(prob_of_actual)
#     avg_log_loss = np.mean(log_loss_vals)
    
#     # Perplexity = exp(Log Loss)
#     # Intuitively: How "confused" is the model? 
#     perplexity = np.exp(avg_log_loss)
    
#     # 5. Calibration Check
#     avg_actual = np.mean(y_test)
#     avg_predicted = np.mean(y_pred_expected)
#     bias = avg_predicted - avg_actual
    
#     # 6. Construct Output Report
#     metrics = {
#         'Metric': [
#             'Mean Absolute Error (MAE)', 
#             'Root Mean Squared Error (RMSE)', 
#             'Log Loss (NLL)', 
#             'Perplexity',
#             'Avg Goals (Actual)', 
#             'Avg Goals (Predicted)', 
#             'Prediction Bias'
#         ],
#         'Value': [mae, rmse, avg_log_loss, perplexity, avg_actual, avg_predicted, bias]
#     }
    
#     results_df = pd.DataFrame(metrics)

#     return results_df

def evaluate_on_test_data(model, X_test, y_test, max_goals=10):
    """
    Generates predictions for the test set, calculates standard errors,
    probabilistic metrics, and the Mean Reciprocal Rank (MRR).
    """
    # 1. Prepare Input & Predict
    X_test_const = sm.add_constant(X_test, has_constant='add')
    y_pred_expected = model.predict(X_test_const)
    
    # 2. Standard Metrics
    mae = mean_absolute_error(y_test, y_pred_expected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_expected))
    
    # 3. Standard Probabilistic Metrics (Log Loss & Perplexity)
    if type(model.model).__name__ == 'NegativeBinomial':
        alpha = model.params[-1]
        n_param = 1.0 / alpha
        p_param = 1.0 / (1.0 + alpha * y_pred_expected)
        prob_of_actual = nbinom.pmf(y_test, n_param, p_param)
    else:
        prob_of_actual = poisson.pmf(y_test, y_pred_expected)
    
    epsilon = 1e-15
    prob_of_actual = np.maximum(prob_of_actual, epsilon)
    log_loss_vals = -np.log(prob_of_actual)
    avg_log_loss = np.mean(log_loss_vals)
    perplexity = np.exp(avg_log_loss)
    
    # 4. Calibration Bias
    avg_actual = np.mean(y_test)
    avg_predicted = np.mean(y_pred_expected)
    bias = avg_predicted - avg_actual
    
    # ==========================================
    # 5. MEAN RECIPROCAL RANK (MRR) INTEGRATION
    # ==========================================
    # Convert to numpy arrays for reliable indexed extraction
    y_pred_arr = np.array(y_pred_expected)
    y_test_arr = np.array(y_test)
    
    # Since data is stacked [All Red Rows, All Blue Rows], matches are offset by N
    num_matches = len(y_test_arr) // 2
    mrr_scores = []
    
    for i in range(num_matches):
        # Extract paired predictions and actuals
        mu_red = y_pred_arr[i]
        mu_blue = y_pred_arr[i + num_matches]
        
        actual_red = y_test_arr[i]
        actual_blue = y_test_arr[i + num_matches]
        
        # Generate probabilities and calculate RR
        prob_array_red = get_pmf_array(mu_red, model, max_goals)
        prob_array_blue = get_pmf_array(mu_blue, model, max_goals)
        
        rr = calculate_match_mrr(prob_array_red, prob_array_blue, actual_red, actual_blue)
        mrr_scores.append(rr)
        
    mrr_final = np.mean(mrr_scores)

    # 6. Construct Output Report
    metrics = {
        'Metric': [
            'Mean Absolute Error (MAE)', 
            'Root Mean Squared Error (RMSE)', 
            'Log Loss (NLL)', 
            'Perplexity',
            'Mean Reciprocal Rank (MRR)', # <--- Appended here
            'Avg Goals (Actual)', 
            'Avg Goals (Predicted)', 
            'Prediction Bias'
        ],
        'Value': [mae, rmse, avg_log_loss, perplexity, mrr_final, avg_actual, avg_predicted, bias]
    }
    
    results_df = pd.DataFrame(metrics)
    return results_df


# def get_score_probability(team_sum, opponent_sum, scaler, model_results):
#     """
#     Calculates the probability of scoring 0 to 10 goals given the skills.
    
#     Args:
#         team_sum (float): Total skill rating of the subject team.
#         opponent_sum (float): Total skill rating of the opponent.
#         scaler (StandardScaler): The scaler fitted on the training data.
#         model_results (statsmodels result): The fitted Poisson model.
        
#     Returns:
#         pd.DataFrame: Probabilities for scoring 0 through 10 goals.
#     """
#     # 1. Calculate Raw Difference
#     raw_diff = team_sum - opponent_sum
    
#     # 2. Normalize the Difference (CRITICAL STEP)
#     # The scaler expects a 2D array, so we reshape the single input
#     norm_diff = scaler.transform(np.array([[raw_diff]])) 
    
#     # 3. Add Constant (Intercept) column for statsmodels
#     # We create an array [[1.0, norm_diff]]
#     exog_input = np.insert(norm_diff, 0, 1.0, axis=1)
    
#     # 4. Predict Expected Goals (Lambda)
#     # The model returns the expected rate (e.g., 8.5 goals)
#     expected_goals = model_results.predict(exog_input)[0]
    
#     # 5. Generate Probability Mass Function (PMF)
#     # Calculate prob for scores 0 to 10
#     scores = np.arange(0, 11)
#     probs = poisson.pmf(scores, expected_goals)
    
#     # 6. Format Output
#     output = pd.DataFrame({
#         'Score': scores,
#         'Probability': np.round(probs * 100, 2) # Convert to percentage
#     })
    
#     return expected_goals, output

def get_latest_player_rating(player_name, df, metric='elo'):
    """
    Finds the most recent rating for a player from the match history.
    """
    # 1. Sort by date to ensure we get the latest game
    # Assuming 'date' column exists and is sortable (strings in ISO format work too)
    # df = df.sort_values(by='date', ascending=True)
    
    # 2. Find rows where the player played in any position
    mask = (
        (df['red defence'] == player_name) | 
        (df['red forward'] == player_name) | 
        (df['blue defence'] == player_name) | 
        (df['blue forward'] == player_name)
    )
    
    player_games = df[mask]
    
    if player_games.empty:
        raise ValueError(f"Player '{player_name}' not found in history.")
        
    # 3. Get the last match they played
    last_match = player_games.iloc[-1]
    
    # 4. Extract rating based on which position they played
    # We check which column matches the player name
    if last_match['red defence'] == player_name:
        return last_match[f'{metric}_red_def']
    elif last_match['red forward'] == player_name:
        return last_match[f'{metric}_red_fwd']
    elif last_match['blue defence'] == player_name:
        return last_match[f'{metric}_blue_def']
    elif last_match['blue forward'] == player_name:
        return last_match[f'{metric}_blue_fwd']
        
    return 1200.0 # Default fallback (should not be reached)

def get_match_score_probabilities(lambda_red, lambda_blue):
    """
    Calculates the exact probability of every final score in a 'First to 10, Win by 2' match.
    
    Args:
        lambda_red (float): Expected goals for Red (from Poisson model).
        lambda_blue (float): Expected goals for Blue (from Poisson model).
        
    Returns:
        pd.DataFrame: Probabilities for all likely final scores (10-8, 11-9, etc.)
    """
    # 1. Determine "Single Goal Probability" (p)
    # p is the probability that the NEXT goal is scored by Red
    if lambda_red + lambda_blue == 0:
        return None # Avoid division by zero
    
    p = lambda_red / (lambda_red + lambda_blue)
    q = 1 - p # Probability Blue scores
    
    # 2. Dynamic Programming Grid for Regulation (Up to 9-9)
    # dp[r][b] = Probability of the score being r-b
    # Size is 10x10 because we stop processing at 9-9 or 10-X
    dp = np.zeros((10, 10))
    dp[0, 0] = 1.0
    
    final_scores = {}
    
    # Iterate through all possible scores up to 9-9
    for r in range(10):
        for b in range(10):
            if r == 9 and b == 9:
                continue # We handle 9-9 separately (The Overtime Gateway)
                
            current_prob = dp[r, b]
            if current_prob == 0:
                continue
            
            # Scenario A: Red Scores Next
            if r + 1 == 10:
                # Red reaches 10. 
                # If Blue < 9, this is a Win (e.g., 10-8). Game Over.
                score_key = f"10-{b}"
                final_scores[score_key] = final_scores.get(score_key, 0) + (current_prob * p)
            else:
                # Game continues
                dp[r + 1, b] += current_prob * p
                
            # Scenario B: Blue Scores Next
            if b + 1 == 10:
                # Blue reaches 10.
                # If Red < 9, this is a Win (e.g., 0-10). Game Over.
                score_key = f"{r}-10"
                final_scores[score_key] = final_scores.get(score_key, 0) + (current_prob * q)
            else:
                # Game continues
                dp[r, b + 1] += current_prob * q

    # 3. Handle Overtime (Starting from 9-9)
    # The probability of reaching 9-9 is stored in dp[9,9]
    prob_reach_ot = dp[9, 9]
    
    # In OT (Win by 2), we simulate rounds of 2 goals.
    # From a tied state (like 9-9, 10-10):
    # - Red Wins (+2): Needs Red, Red (p*p)
    # - Blue Wins (+2): Needs Blue, Blue (q*q)
    # - Tie Continues: Needs Red,Blue OR Blue,Red (2*p*q)
    
    prob_red_win_round = p * p
    prob_blue_win_round = q * q
    prob_tie_round = 2 * p * q
    
    # We calculate probable OT scores up to a reasonable limit (e.g., 15-13)
    # k represents the "round" of overtime (0 = 11-9, 1 = 12-10, etc)
    current_tie_prob = prob_reach_ot
    
    for k in range(10): # Limit loop to avoid infinite running, cover up to ~20 goals
        # Calculate scores for this round
        red_winning_score = 11 + k
        red_losing_score = 9 + k
        blue_winning_score = 11 + k
        blue_losing_score = 9 + k
        
        # Red wins this round
        prob_red_win_now = current_tie_prob * prob_red_win_round
        final_scores[f"{red_winning_score}-{red_losing_score}"] = prob_red_win_now
        
        # Blue wins this round
        prob_blue_win_now = current_tie_prob * prob_blue_win_round
        final_scores[f"{blue_losing_score}-{blue_winning_score}"] = prob_blue_win_now
        
        # Update tie probability for next round (e.g., going from 10-10 to 11-11)
        current_tie_prob = current_tie_prob * prob_tie_round
        
        # Optimization: Stop if probability is negligible
        if current_tie_prob < 0.0001: 
            break

    # 4. Format Output
    # Convert dict to DataFrame
    scores_list = list(final_scores.keys())
    probs_list = list(final_scores.values())
    
    result_df = pd.DataFrame({
        'Score': scores_list,
        'Probability': probs_list
    })
    
    # Sort: First by Winner (Red then Blue), then by Margin
    # Helper to parse scores for sorting
    def parse_score(s):
        r, b = map(int, s.split('-'))
        return r, b

    result_df['R'], result_df['B'] = zip(*result_df['Score'].map(parse_score))
    result_df['Total_Prob'] = result_df['Probability'] * 100 # for display
    
    # Sort logic: Red wins descending margin, then Blue wins ascending margin
    result_df = result_df.sort_values(by=['R', 'B'], ascending=[False, True])
    
    return result_df[['Score', 'Probability']]

# --- Updated Wrapper ---
def predict_match_full_detail(r_def, r_fwd, b_def, b_fwd, raw_df, scaler, model, metric):
    """
    Wrapper that predicts specific score probabilities (e.g., 10-8).
    """
    try:
        # Lookup and sums
        r_def_skill = get_latest_player_rating(r_def, raw_df, metric)
        r_fwd_skill = get_latest_player_rating(r_fwd, raw_df, metric)
        b_def_skill = get_latest_player_rating(b_def, raw_df, metric)
        b_fwd_skill = get_latest_player_rating(b_fwd, raw_df, metric)
        
        red_sum = r_def_skill + r_fwd_skill
        blue_sum = b_def_skill + b_fwd_skill
        
        # Get Poisson Lambdas (Expected Goals)
        # We need the lambda to calculate 'p' (Red's relative strength)
        
        # 1. Red Lambda
        raw_diff_red = red_sum - blue_sum
        norm_diff_red = scaler.transform(np.array([[raw_diff_red]])) 
        exog_red = np.insert(norm_diff_red, 0, 1.0, axis=1)
        lambda_red = model.predict(exog_red)[0]
        
        # 2. Blue Lambda
        raw_diff_blue = blue_sum - red_sum
        norm_diff_blue = scaler.transform(np.array([[raw_diff_blue]]))
        exog_blue = np.insert(norm_diff_blue, 0, 1.0, axis=1)
        lambda_blue = model.predict(exog_blue)[0]
        
        print(f"Players: {r_def} DEF/{r_fwd} FWD (Red) VS {BLUE_DEFENCE} DEF/{BLUE_FORWARD} FWD (Blue)")
        print(f"Matchup: Red ({red_sum}) VS Blue ({blue_sum})")
        print(f"Scoring Rates: Red={lambda_red:.2f}, Blue={lambda_blue:.2f}")
        
        # Call the new Matrix Engine
        score_probs = get_match_score_probabilities(lambda_red, lambda_blue)
        
        # Display Top Outcomes
        print("\nMost Likely Final Scores:")
        # Format as percentage
        score_probs['Probability'] = (score_probs['Probability'] * 100).round(2)
        print(score_probs.sort_values('Probability', ascending=False).head(10).to_string(index=False))
        
        # Sum probabilities to see Win Chance
        red_wins = score_probs[score_probs['Score'].apply(lambda x: int(x.split('-')[0]) > int(x.split('-')[1]))]
        print(f"\nTotal Red Win Probability: {red_wins['Probability'].sum():.2f}%")
        print(f"Total Blue Win Probability: {100-red_wins['Probability'].sum():.2f}%")
        
    except ValueError as e:
        print(e)


## MRR CALCULATION ------------
def get_pmf_array(mu, model_obj, max_goals=10):
    """Generates the PMF array for a specific expected goal parameter (mu)."""
    k_vals = np.arange(max_goals + 1)
    
    if type(model_obj.model).__name__ == 'NegativeBinomial':
        alpha = model_obj.params[-1]
        n_param = 1.0 / alpha
        p_param = 1.0 / (1.0 + alpha * mu)
        return nbinom.pmf(k_vals, n_param, p_param)
    else:
        return poisson.pmf(k_vals, mu)

def calculate_match_mrr(prob_array_red, prob_array_blue, actual_red, actual_blue):
    """Calculates the Reciprocal Rank for a single match."""
    joint_probs = np.outer(prob_array_red, prob_array_blue)
    
    flat_probs = []
    for r in range(joint_probs.shape[0]):
        for b in range(joint_probs.shape[1]):
            flat_probs.append((joint_probs[r, b], r, b))
            
    flat_probs.sort(key=lambda x: x[0], reverse=True)
    
    for rank, (prob, r, b) in enumerate(flat_probs, start=1):
        if r == actual_red and b == actual_blue:
            return 1.0 / rank
            
    return 0.0


def evaluate_monthly_mrr(model_red, model_blue, X_test_red, X_test_blue, y_test_red, y_test_blue, max_goals=10):
    """
    Wrapper to evaluate the Mean Reciprocal Rank across an entire test month.
    """
    # 1. Prepare features and predict Lambdas for the whole test set
    X_test_red_const = sm.add_constant(X_test_red, has_constant='add')
    X_test_blue_const = sm.add_constant(X_test_blue, has_constant='add')
    
    lambdas_red = model_red.predict(X_test_red_const)
    lambdas_blue = model_blue.predict(X_test_blue_const)
    
    mrr_scores = []
    
    # 2. Iterate through every match in the test set
    # Using enumerate and .iloc to safely handle pandas Series indices
    for i in range(len(y_test_red)):
        mu_red = lambdas_red.iloc[i] if hasattr(lambdas_red, 'iloc') else lambdas_red[i]
        mu_blue = lambdas_blue.iloc[i] if hasattr(lambdas_blue, 'iloc') else lambdas_blue[i]
        
        actual_red = y_test_red.iloc[i] if hasattr(y_test_red, 'iloc') else y_test_red[i]
        actual_blue = y_test_blue.iloc[i] if hasattr(y_test_blue, 'iloc') else y_test_blue[i]
        
        # 3. Generate probability distributions for this specific match
        prob_array_red = get_pmf_array(mu_red, model_red, max_goals)
        prob_array_blue = get_pmf_array(mu_blue, model_blue, max_goals)
        
        # 4. Calculate RR for this match
        rr = calculate_match_mrr(prob_array_red, prob_array_blue, actual_red, actual_blue)
        mrr_scores.append(rr)
        
    # 5. Return the Mean Reciprocal Rank
    return np.mean(mrr_scores)


# --- Main Execution Block ---
if __name__ == "__main__":
    # path to your dataset
    file_path = './data/rankings.xlsx' 
    METRIC = "vitelo" #ts_mu

    # predict test variables
    RED_DEFENCE = "Guillermo"
    RED_FORWARD = "Seyda"
    BLUE_DEFENCE = "RiccardoP"
    BLUE_FORWARD = "Vito"


    # # historic matches rankings data
    historic_df = pd.read_excel(file_path)
    
    # print("\n-------------- SINGLE SPLIT DEMO ------------")
    # print(f"\n--- Starting Data Processing with rating: {METRIC} ---")

    # train_df, test_df = transform_foosball_data(historic_df, metric=METRIC)

    # # print("\nModeling Data Statistics:")
    # # print(train_df.describe())
    
    # print("\n-------------- Data split phase ------------")

    # X_train, X_test, y_train, y_test, scaler = split_and_normalize(train_df, test_df)
    
    # print(f"Successfully loaded and stacked data.")
    # print(f"Total observations for training: {len(X_train)}")
    
    # # training splits
    # # print(X_train)
    # # print(X_test)
    # print("\n-------------- TRAINING phase ------------")
    # trained_models, model_metrics = train_glms(X_train, y_train)

    # # print("\n")
    # # print(trained_models)
    # # print(model_metrics)

    # print("\n-------------- Evaluation phase ------------")

    # results_df = evaluate_on_test_data(trained_models['Poisson'], X_test, y_test)
    # print(f"--- Test Set Evaluation: Poison ---")
    # print(results_df.to_string(index=False, float_format="%.4f"))

    # results_df = evaluate_on_test_data(trained_models['NegativeBinomial'], X_test, y_test)
    # print(f"--- Test Set Evaluation: NegativeBinomialoison ---")
    # print(results_df.to_string(index=False, float_format="%.4f"))


    # print("\n-------------- PREDICTION DEMO phase ------------")

    # match_result = predict_match_full_detail(
    #         r_def=RED_DEFENCE, r_fwd=RED_FORWARD, 
    #         b_def=BLUE_DEFENCE, b_fwd=BLUE_FORWARD,
    #         raw_df=historic_df,
    #         scaler=scaler,
    #         model=trained_models['Poisson'],
    #         metric=METRIC
    #     )
        
    print(f"\n-------------- CROSS VALIDATION EXPERIMENT FOR METRIC: {METRIC}------------")
    # CROSS VALIDATION EXPERIMENT
    # ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01"]
    month_results_poi = []
    month_results_nbm = []
    for month in ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01"]:

        print(f"\n------ EVALUATING IN MONTH: {month} ------")
        train_df, test_df = transform_foosball_data(historic_df, metric=METRIC, test_period=month)

        X_train, X_test, y_train, y_test, scaler = split_and_normalize(train_df, test_df)

        trained_models, model_metrics = train_glms(X_train, y_train)

        results_df_poi = evaluate_on_test_data(trained_models['Poisson'], X_test, y_test)
        results_df_nbm = evaluate_on_test_data(trained_models['NegativeBinomial'], X_test, y_test)

        month_results_poi.append(results_df_poi)
        month_results_nbm.append(results_df_nbm)

    print("\n" + "="*40)
    print("   FINAL CROSS-VALIDATION RESULTS")
    print("="*40)
    
    # Concatenate all fold results into one DataFrame
    all_results_poi = pd.concat(month_results_poi, ignore_index=True)
    all_results_nbm = pd.concat(month_results_nbm, ignore_index=True)
    
    print("POISSON MODEL")
    # Group by 'Metric' and calculate the mean of 'Value'
    # This averages the MAE, RMSE, Log Loss, etc. across all 5 months
    final_averages_poi = all_results_poi.groupby('Metric')['Value'].mean().reset_index()
    # Sort for readability (Optional)
    # We can map specific sort order if desired, otherwise alphabetical
    print(final_averages_poi.to_string(index=False, float_format="%.4f"))

    print("-" * 40)
    print("NBM MODEL")

    final_averages_nbm = all_results_nbm.groupby('Metric')['Value'].mean().reset_index()
    
    # Sort for readability (Optional)
    # We can map specific sort order if desired, otherwise alphabetical
    print(final_averages_nbm.to_string(index=False, float_format="%.4f"))
    
    print("-" * 40)
    # print(f"Metrics averaged over {len(month_results)} folds.")

#### NOTES 
# to implement:
# train the models directly with the rankings, no team diff
# train the model with score difference as well
# include 10-10 scores as a anomalous case, extended
# filter matches by overtime and see if they are over-represented by doing the same experiments
# compute likelihood for the win, based on the model fit on data
# mean reciprocal rank for measuring if score is in the predictions
#-- extra research
# interest question: how the system responds to innovation? players discovers new skills or new players come and change the dinamics
# iq2: 
# test: given a small set of players: compute elos per pair, in config, and repeat the experiment, check if a strong attacker enhances the defence, or viceversa
# randomness??