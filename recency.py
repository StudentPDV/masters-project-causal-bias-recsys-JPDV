import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import time

dataset = 'm32'
DATA_PATH = f'data_recency.pkl'

def calculate_propensity_score(df, confounder_cols, treatment_col):
    X = df[confounder_cols]
    y = df[treatment_col]
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]


def run_ipw(df, treatment_col, outcome_col, ps_col):
    T = df[treatment_col]
    Y = df[outcome_col]
    ps = np.clip(df[ps_col], 0.001, 0.999)
    return np.mean((T * Y / ps) - ((1 - T) * Y / (1 - ps)))


def run_psm(df, treatment_col, outcome_col, ps_col):
    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)
    if len(treated) == 0 or len(control) == 0: return np.nan

    control_ps = control[[ps_col]].values
    treated_ps = treated[[ps_col]].values

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_ps)
    distances, indices = nbrs.kneighbors(treated_ps)

    matched_outcomes = control.iloc[indices.flatten()][outcome_col].values
    return np.mean(treated[outcome_col].values - matched_outcomes)


def main():
    print(f"--- RECENCY BIAS EXPERIMENT ---")
    print(f"1. Loading Data...")
    df = pd.read_pickle(DATA_PATH)

    T_COL = 'T_rec'  
    C_COLS = ['avg_rating']  

    print(f"2. Calculating Propensity Scores (T={T_COL})...")
    start_time = time.time()
    df['ps'] = calculate_propensity_score(df, C_COLS, T_COL)
    print(f"   Done in {time.time() - start_time:.2f}s")

    experiments = [
        ("EXP-09", "Raw Ratings", "IPW", 'Y_raw', run_ipw),
        ("EXP-10", "Raw Ratings", "PSM", 'Y_raw', run_psm),
        ("EXP-11", "Normalized", "IPW", 'Y_norm', run_ipw),
        ("EXP-12", "Normalized", "PSM", 'Y_norm', run_psm),
    ]

    print(f"\n3. Running Results...")
    print(f"{'ID':<8} | {'Condition':<12} | {'Method':<6} | {'ATE':<10} | {'Time (s)':<10}")
    print("-" * 60)

    results = []
    for exp_id, cond, method_name, y_col, func in experiments:
        t0 = time.time()
        ate = func(df, T_COL, y_col, 'ps')
        runtime = time.time() - t0
        print(f"{exp_id:<8} | {cond:<12} | {method_name:<6} | {ate:>8.4f}   | {runtime:>8.2f}")
        results.append({'Exp': exp_id, 'Condition': cond, 'Method': method_name, 'ATE': ate, 'Runtime': runtime})

    pd.DataFrame(results).to_csv('recency_bias_results.csv', index=False)
    print("\nSaved to recency_bias_results.csv")


if __name__ == "__main__":
    main()
