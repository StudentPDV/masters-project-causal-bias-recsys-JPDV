import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import time

dataset = 'm32'
DATA_PATH = 'data_popularity.pkl'

def calculate_propensity_score(df, confounder_cols, treatment_col):
    X = df[confounder_cols]
    y = df[treatment_col]

    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X, y)

    ps_scores = clf.predict_proba(X)[:, 1]
    return ps_scores


def run_ipw(df, treatment_col, outcome_col, ps_col):
    T = df[treatment_col]
    Y = df[outcome_col]
    ps = df[ps_col]

    ps = np.clip(ps, 0.001, 0.999)
    ate = np.mean((T * Y / ps) - ((1 - T) * Y / (1 - ps)))
    return ate


def run_psm(df, treatment_col, outcome_col, ps_col):
    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)

    if len(treated) == 0 or len(control) == 0:
        return np.nan

    control_ps = control[[ps_col]].values
    treated_ps = treated[[ps_col]].values

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_ps)

    distances, indices = nbrs.kneighbors(treated_ps)

    matched_control_indices = indices.flatten()
    matched_outcomes = control.iloc[matched_control_indices][outcome_col].values

    differences = treated[outcome_col].values - matched_outcomes
    ate = np.mean(differences)

    return ate


def main():
    print(f"--- 1. LOADING DATA ---")
    df = pd.read_pickle(DATA_PATH)

    print(f"   Data Loaded. Rows: {len(df)}")

    T_COL = 'T_pop'  
    C_COLS = ['avg_rating']  

    print(f"\n--- 2. PROPENSITY SCORING ---")
    start_time = time.time()
    df['ps'] = calculate_propensity_score(df, C_COLS, T_COL)
    print(f"   Propensity Scores calculated in {time.time() - start_time:.2f}s")

    results = []

    experiments = [
        ("EXP-01", "Raw Ratings", "IPW", 'Y_raw', run_ipw),
        ("EXP-02", "Raw Ratings", "PSM", 'Y_raw', run_psm),
        ("EXP-03", "Normalized", "IPW", 'Y_norm', run_ipw),
        ("EXP-04", "Normalized", "PSM", 'Y_norm', run_psm),
    ]

    print(f"\n--- 3. RUNNING EXPERIMENTS ---")
    print(f"{'ID':<8} | {'Condition':<12} | {'Method':<6} | {'ATE':<10} | {'Time (s)':<10}")
    print("-" * 60)

    for exp_id, cond, method_name, y_col, func in experiments:
        t0 = time.time()

        ate_est = func(df, T_COL, y_col, 'ps')

        runtime = time.time() - t0
        print(f"{exp_id:<8} | {cond:<12} | {method_name:<6} | {ate_est:>8.4f}   | {runtime:>8.2f}")

        results.append({
            'Experiment': exp_id,
            'Condition': cond,
            'Method': method_name,
            'ATE': ate_est,
            'Runtime': runtime
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv('popularity_bias_results.csv', index=False)
    print("\n--- DONE! Results saved to popularity_bias_results.csv ---")


if __name__ == "__main__":
    main()
