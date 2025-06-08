import numpy as np
import pandas as pd
from math import sqrt


def ahp_process(comparison_matrix):
    """
    Process the AHP comparison matrix to get weights
    """
    # Convert to numpy array for calculations
    matrix = np.array(comparison_matrix, dtype=float)
    n = len(matrix)
    
    # Calculate column sums
    col_sums = matrix.sum(axis=0)
    
    # Normalize the matrix
    normalized_matrix = matrix / col_sums
    
    # Calculate weights (row averages)
    weights = normalized_matrix.mean(axis=1)
    
    # Check consistency
    # Calculate lambda_max (principal eigenvalue)
    weighted_sum = np.dot(matrix, weights)
    consistency_vector = weighted_sum / weights
    lambda_max = np.mean(consistency_vector)
    
    # Calculate Consistency Index (CI)
    if n > 1:
        ci = (lambda_max - n) / (n - 1)
    else:
        ci = 0  # Or handle as a specific case, e.g., None, or -1
    
    # Random Consistency Index (RI) values for n=1 to n=10
    ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_values.get(n, 1.5)  # Default to 1.5 for larger matrices
    
    # Calculate Consistency Ratio (CR)
    cr = ci / ri if ri != 0 else 0
    
    return weights, cr

def profile_matching(alternatives, sub_criteria_weights):
    # gap method 
    scores = []
    for alt in alternatives:
        total_score = 0
        for main, subs in sub_criteria_weights.items():
            for sub, weight in subs.items():
                total_score += alt[sub] * weight
        scores.append(total_score)
    return scores

def calculate_profile_matching_score(monitor_values, ideal_values, gap_weights):
    """
    Calculate profile matching score for each criterion
    """
    gap = monitor_values - ideal_values
    score = gap_weights[gap]
    return score

def topsis(decision_matrix, weights):
    """
    Implement TOPSIS method for ranking alternatives
    """
    # Step 1: Normalize the decision matrix
    criteria = decision_matrix.columns.tolist()
    normalized_matrix = decision_matrix.copy()
    
    for criterion in criteria:
        # Calculate the square root of the sum of squares for each criterion
        denominator = sqrt(sum(decision_matrix[criterion] ** 2))
        if denominator != 0:  # Prevent division by zero
            normalized_matrix[criterion] = decision_matrix[criterion] / denominator
        else:
            normalized_matrix[criterion] = 0
    
    # Step 2: Calculate the weighted normalized decision matrix
    weighted_matrix = normalized_matrix.copy()
    for criterion in criteria:
        weighted_matrix[criterion] = normalized_matrix[criterion] * weights[criterion]
    
    # Step 3: Determine the ideal and negative-ideal solutions
    ideal_solution = {}
    negative_ideal_solution = {}
    
    for criterion in criteria:
        ideal_solution[criterion] = weighted_matrix[criterion].max()
        negative_ideal_solution[criterion] = weighted_matrix[criterion].min()
    
    # Step 4: Calculate separation measures from ideal and negative-ideal solutions
    separation_positive = {}
    separation_negative = {}
    
    for monitor in weighted_matrix.index:
        separation_positive[monitor] = sqrt(sum((weighted_matrix.loc[monitor] - pd.Series(ideal_solution)) ** 2))
        separation_negative[monitor] = sqrt(sum((weighted_matrix.loc[monitor] - pd.Series(negative_ideal_solution)) ** 2))
    
    # Step 5: Calculate the relative closeness to the ideal solution
    closeness = {}
    for monitor in weighted_matrix.index:
        # closeness[monitor] = separation_negative[monitor] / (separation_positive[monitor] + separation_negative[monitor])
        denominator = separation_positive[monitor] + separation_negative[monitor]
        if denominator == 0:
            closeness[monitor] = 0  # or maybe 0.5 if both separations are 0 and you want a neutral closeness
        else:
            closeness[monitor] = separation_negative[monitor] / denominator

    # Convert to DataFrame for better visualization
    closeness_df = pd.DataFrame(list(closeness.items()), columns=['Monitor', 'Closeness'])
    closeness_df = closeness_df.sort_values('Closeness', ascending=False)
    
    return closeness_df

def create_categorical_dataframe(df):
    import pandas as pd

    categorical_df = pd.DataFrame()

    categorization_rules = {
        'Refresh Rate': {
            (0, 76): 1,
            (76, 145): 2,
            (145, 241): 3,
            (241, 361): 4,
            (361, float('inf')): 5
        },
        'Resolution': {
            "HD (1280 x 720)": 1,
            "Full HD (1920 x 1080)": 2,
            "Ultra HD (2560 x 1440)": 3,
            "Quad HD (3840 x 2160)": 4,
            "8K (7680 x 4320)": 5
        },
        'Screen type': {
            'OLED': 5,
            'IPS': 3,
            'TN': 1,
            'VA': 1,
        },
        'Size': {
            (0, 20): 1,
            (20, 25): 2,
            (25, 30): 3,
            (30, 35): 4,
            (35, float('inf')): 5
        },
        'Weight': {
            (0, 5): 1,
            (5, 7): 2,
            (7, 9): 3,
            (9, 11): 4,
            (11, float('inf')): 5
        },
        'Price': {
            (0, 2000001): 1,
            (2000001, 4000001): 2,
            (4000001, 6000001): 3,
            (6000001, 8000001): 4,
            (8000001, float('inf')): 5
        },
        'Warranty': {
            (0, 2): 1,
            (2, 4): 2,
            (4, 6): 3,
            (6, 8): 4,
            (8, float('inf')): 5
        },
    }

    for column, rules in categorization_rules.items():
        if column not in df.columns:
            raise ValueError(f"Missing expected column: {column}")
        
        if isinstance(list(rules.keys())[0], tuple):
            bins = sorted([r[0] for r in rules.keys()] + [list(rules.keys())[-1][1]])
            labels = [rules[k] for k in sorted(rules.keys())]
            categorical_df[column] = pd.cut(df[column], bins=bins, labels=labels, right=False).astype(int)
        else:
            # Round float to int if necessary
            if df[column].dtype == float:
                df[column] = df[column].round().astype(int)
            categorical_df[column] = df[column].map(rules)

    # Handle Features
    if 'Features' in df.columns:
        categorical_df['Features'] = df['Features'].astype(int)

    return categorical_df
