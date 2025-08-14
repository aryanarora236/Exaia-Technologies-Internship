#!/usr/bin/env python3
"""
cymo_analysis_with_tuning_top10_all_features.py

1. Load sentence-level CYMO CSVs (authentic & synthetic).
2. Extract user IDs from TID (prefix before underscore).
3. Aggregate (mean) features per user.
4. Sample 5400 users from each kind to balance.
5. Split users into train/dev/test (80/10/10 stratified).
6. Grid-search hyperparameters using train/dev.
7. Train final model on train+dev, evaluate on test.
8. UMAP visualization of all users.
9. Compute JS divergence & Wasserstein distance for each aggregated feature (all ~344 features),
   but only save density plots for a selected subset of 8 features.
10. Save density plots, summary CSV, and top-10 lists for each metric over all features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, ParameterGrid
import umap

# ─── CONFIG ───────────────────────────────────────────────────────────────────
AUTH_PATH   = "IMPORTANT files/CYMO_Outputs/AUTHENTIC_USING_DATASETS/ann.balanced_control_part1.csv"
SYN_PATH    = "IMPORTANT files/CYMO_Outputs/FINAL/SYNTHETIC CONTROL/ann.formatted_finalversion_control_5400_inferred_few_shot.csv"
FIG_DIR     = "IMPORTANT files/AnalysisResults/REAL_FINAL/final_synthetic_control/figures_IFS/"
SAMPLE_SIZE = 3575

os.makedirs(FIG_DIR, exist_ok=True)

# Columns to exclude from feature calculations
NON_FEATURE_COLS = ["TID", "sid", "label", "source", "model"]

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def extract_user_id(df):
    df = df.copy()
    df['user_id'] = df['TID'].astype(str).str.split('_').str[0]
    return df


def load_and_aggregate(path, kind_label):
    df = pd.read_csv(path)
    df = extract_user_id(df)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + ['TID', 'user_id']]
    agg = df.groupby('user_id')[feat_cols].mean().reset_index()
    agg['kind'] = kind_label
    return agg

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1) Load & aggregate
    auth_users = load_and_aggregate(AUTH_PATH, 'authentic')
    syn_users  = load_and_aggregate(SYN_PATH,  'synthetic')
    print("Authentic:", auth_users.shape[0], "rows; unique users:", auth_users['user_id'].nunique())
    print("Synthetic:", syn_users.shape[0], "rows; unique users:", syn_users['user_id'].nunique())

    # 2) Sample to balance
    auth_ids = auth_users['user_id'].unique()[:SAMPLE_SIZE]
    syn_ids  = syn_users['user_id'].unique()[:SAMPLE_SIZE]
    auth_users = auth_users[auth_users['user_id'].isin(auth_ids)].reset_index(drop=True)
    syn_users  = syn_users[syn_users['user_id'].isin(syn_ids)].reset_index(drop=True)

    # 3) Combine and prepare data
    users = pd.concat([auth_users, syn_users], ignore_index=True)
    feat_cols = [c for c in users.columns if c not in ['user_id', 'kind']]
    X_all = users[feat_cols]
    y_all = (users['kind'] == 'synthetic').astype(int)

    # 4) Train/dev/test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    # 5) Hyperparameter tuning on train/dev
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    best_score, best_params = -np.inf, None
    for params in ParameterGrid(param_grid):
        clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        clf.fit(X_train, y_train)
        y_dev_pred = clf.predict(X_dev)
        score = f1_score(y_dev, y_dev_pred)
        if score > best_score:
            best_score, best_params = score, params
    print(f"Best params (dev F1={best_score:.3f}): {best_params}")

    # 6) Train final model on train+dev
    X_train_dev = pd.concat([X_train, X_dev], ignore_index=True)
    y_train_dev = pd.concat([y_train, y_dev], ignore_index=True)
    final_clf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    final_clf.fit(X_train_dev, y_train_dev)

    # 7) Evaluate on test set
    y_test_pred = final_clf.predict(X_test)
    y_test_prob = final_clf.predict_proba(X_test)[:, 1]
    print("\n=== Test set classification ===")
    print(classification_report(y_test, y_test_pred,
                                target_names=['authentic','synthetic']))
    print("Test AUC:", roc_auc_score(y_test, y_test_prob))

    # 8) UMAP visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(X_all)
    df_umap = pd.DataFrame(emb, columns=['UMAP1', 'UMAP2'])
    df_umap['kind'] = users['kind']
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='kind', data=df_umap, alpha=0.7)
    plt.title('UMAP of user-level CYMO features')
    umap_path = os.path.join(FIG_DIR, "umap_users.png")
    plt.savefig(umap_path, dpi=150)
    plt.close()
    print(f"Saved UMAP plot to: {umap_path}")

    # 9) Distributional distances for all features + KDE for selected ones
    ALL_FEATURES = [c for c in auth_users.columns if c not in ['user_id', 'kind']]
    DENSITY_FEATURES = [
        'MLS', 'CT', 'NDW', 'TTR', 'cTTR', 'LD', 'EMOanx', 'EMOsad'
    ]
    results = []
    for feat in ALL_FEATURES:
        a = auth_users[feat].dropna()
        s = syn_users[feat].dropna()

        # compute histograms for JS
        a_hist, bins = np.histogram(a, bins=50, density=True)
        s_hist, _    = np.histogram(s, bins=bins, density=True)
        eps = 1e-12
        a_hist += eps; s_hist += eps
        a_hist /= a_hist.sum(); s_hist /= s_hist.sum()

        js = jensenshannon(a_hist, s_hist, base=2)
        wd = wasserstein_distance(a, s)
        results.append({'feature': feat, 'JS_divergence': js, 'Wasserstein': wd})

        if feat in DENSITY_FEATURES:
            plt.figure(figsize=(6,4))
            sns.kdeplot(a, label='authentic', fill=True, alpha=0.5)
            sns.kdeplot(s, label='synthetic', fill=True, alpha=0.5)
            plt.title(f"{feat}  JS={js:.3f}  WD={wd:.3f}")
            plt.xlabel(feat)
            plt.legend()
            density_path = os.path.join(FIG_DIR, f"density_{feat}.png")
            plt.savefig(density_path, dpi=150)
            plt.close()
            print(f"Saved density plot for {feat} to: {density_path}")

    # 10) Summaries and top-10 lists over all features
    df_res = pd.DataFrame(results).set_index('feature')
    csv_path = os.path.join(FIG_DIR, "feature_distance_summary.csv")
    df_res.to_csv(csv_path)
   # print("\n=== Distributional distances (all features) ===")
   # print(df_res.to_string())
    print(f"Saved distance summary CSV to: {csv_path}")

    top10_js = df_res.sort_values('JS_divergence', ascending=False).head(10)
    top10_ws = df_res.sort_values('Wasserstein',   ascending=False).head(10)

    print("\n=== Top 10 by JS divergence ===")
    print(top10_js.to_string())
    print("\n=== Top 10 by Wasserstein distance ===")
    print(top10_ws.to_string())

    js_csv = os.path.join(FIG_DIR, "top10_js_divergence.csv")
    ws_csv = os.path.join(FIG_DIR, "top10_wasserstein_distance.csv")
    top10_js.to_csv(js_csv)
    top10_ws.to_csv(ws_csv)
    print(f"Saved top-10 JS to: {js_csv}")
    print(f"Saved top-10 WS to: {ws_csv}\n")
