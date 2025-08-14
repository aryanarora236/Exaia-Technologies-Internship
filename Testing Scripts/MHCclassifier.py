#!/usr/bin/env python3
'''
cymo_user_rf_augment.py

1. Load sentence-level CYMO CSVs:
     - authentic positive
     - authentic control
     - synthetic positive
     - synthetic control
2. Extract user_id from TID.
3. Keep only the first N unique users per dataset.
4. Aggregate (mean) features per user.
5. Split authentic users into train/dev/test (80/10/10 stratified).
6. Augment **training** with fractions of authentic positive, authentic control, synthetic positive, and synthetic control.
7. Grid‐search RF on train/dev (maximize user‐level F1).
8. Train final RF on train+dev (augmented), evaluate on test.
9. UMAP + distributional analyses on aggregated authentic user features.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, ParameterGrid
import umap

# ─── USER SETTINGS ─────────────────────────────────────────────────────────────
AUTH_POS_PATH     = 'IMPORTANT files/CYMO_Outputs/AUTHENTIC_USING_DATASETS/Combined_CYMO_bipolar.csv'
AUTH_CTRL_PATH    = 'IMPORTANT files/CYMO_Outputs/AUTHENTIC_USING_DATASETS/ann.balanced_control_part1.csv'
SYN_POS_PATH      = 'IMPORTANT files/CYMO_Outputs/FINAL/SYNTHETIC POSITIVE/ann.formatted_final5400_inferred_few_shot.csv'
SYN_CTRL_PATH     = 'IMPORTANT files/CYMO_Outputs/FINAL/SYNTHETIC CONTROL/ann.formatted_finalversion_control_5400_inferred_few_shot.csv'
FIG_DIR           = 'AnalysisResults/REAL_FINAL/figures_user_augment_IFS_with0.5synthetic/'
os.makedirs(FIG_DIR, exist_ok=True)

# Total unique users to take from each raw dataset
TOTAL_USERS       = 5400

# Fractions to include in train augmentation (0.0–1.0)
AUTH_POS_PCT      = 1.0   # fraction of authentic positive users to include
AUTH_CTRL_PCT     = 1.0   # fraction of authentic control users to include
AUG_SYN_POS_PCT   = 1.0     # fraction of synthetic positive users to include
AUG_SYN_CTRL_PCT  = 1.0     # fraction of synthetic control users to include

# RF hyperparameter grid (user‐level)
PARAM_GRID = {
    'n_estimators':      [50, 100],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2]
}

# Columns to drop when building feature matrices
NON_FEATURE_COLS = ['TID', 'sid', 'label', 'source', 'model']

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def extract_user_id(df):
    df = df.copy()
    df['user_id'] = df['TID'].astype(str).str.split('_').str[0]
    return df


def limit_users(df, max_users):
    uids = df['user_id'].unique()[:max_users]
    return df[df['user_id'].isin(uids)].copy()


def aggregate_user_features(df):
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS + ['TID', 'sid', 'user_id', 'label']]
    agg = df.groupby('user_id')[feat_cols].mean().reset_index()
    return agg, feat_cols


def plot_umap(features, labels, fname):
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(features)
    dfu = pd.DataFrame(emb, columns=['UMAP1', 'UMAP2'])
    dfu['label'] = labels.values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='label', data=dfu, alpha=0.7)
    plt.title('UMAP of Authentic Users (control vs positive)')
    plt.savefig(fname, dpi=150)
    plt.close()


def compute_and_plot_dists(ctrl_df, pos_df, feat_cols):
    results = []
    for feat in feat_cols:
        a = ctrl_df[feat].dropna()
        p = pos_df[feat].dropna()
        ah, edges = np.histogram(a, bins=50, density=True)
        ph, _ = np.histogram(p, bins=edges, density=True)
        eps = 1e-12
        ah += eps; ph += eps
        ah /= ah.sum(); ph /= ph.sum()
        js = jensenshannon(ah, ph, base=2)
        wd = wasserstein_distance(a, p)
        plt.figure(figsize=(6, 4))
        sns.kdeplot(a, label='control', fill=True, alpha=0.5)
        sns.kdeplot(p, label='positive', fill=True, alpha=0.5)
        plt.title(f"{feat} — JS={js:.3f}, WD={wd:.3f}")
        plt.xlabel(feat)
        plt.legend()
        plt.savefig(os.path.join(FIG_DIR, f"density_{feat}.png"), dpi=150)
        plt.close()
        results.append({'feature': feat, 'JS_divergence': js, 'Wasserstein': wd})
    dfres = pd.DataFrame(results).set_index('feature')
    dfres.to_csv(os.path.join(FIG_DIR, 'feature_distance_summary.csv'))
    return dfres

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1) Load, extract user_id, limit by TOTAL_USERS
    df_pos_auth  = pd.read_csv(AUTH_POS_PATH).pipe(extract_user_id)
    df_pos_auth  = limit_users(df_pos_auth, TOTAL_USERS)
    df_pos_auth['label'] = 'positive'

    df_ctrl_auth = pd.read_csv(AUTH_CTRL_PATH).pipe(extract_user_id)
    df_ctrl_auth = limit_users(df_ctrl_auth, TOTAL_USERS)
    df_ctrl_auth['label'] = 'control'

    df_syn_pos   = pd.read_csv(SYN_POS_PATH).pipe(extract_user_id)
    df_syn_pos   = limit_users(df_syn_pos, TOTAL_USERS)
    df_syn_pos['label'] = 'positive'

    df_syn_ctrl  = pd.read_csv(SYN_CTRL_PATH).pipe(extract_user_id)
    df_syn_ctrl  = limit_users(df_syn_ctrl, TOTAL_USERS)
    df_syn_ctrl['label'] = 'control'

    # 2) Aggregate
    agg_pos_auth, feat_cols = aggregate_user_features(df_pos_auth)
    agg_pos_auth['label'] = 'positive'

    agg_ctrl_auth, _        = aggregate_user_features(df_ctrl_auth)
    agg_ctrl_auth['label'] = 'control'

    agg_syn_pos, _          = aggregate_user_features(df_syn_pos)
    agg_syn_pos['label'] = 'positive'

    agg_syn_ctrl, _         = aggregate_user_features(df_syn_ctrl)
    agg_syn_ctrl['label'] = 'control'

    # 3) Prepare authentic
    auth_agg = pd.concat([agg_ctrl_auth, agg_pos_auth], ignore_index=True)
    auth_agg['y'] = auth_agg['label'].map({'control': 0, 'positive': 1})

    # 4) Split authentic 80/10/10 stratified
    uids_auth = auth_agg['user_id'].values
    y_auth    = auth_agg['y'].values
    u_train, u_tmp, y_tr, y_tmp = train_test_split(
        uids_auth, y_auth, test_size=0.20, stratify=y_auth, random_state=42)
    u_dev, u_test, y_dev, y_test = train_test_split(
        u_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    train_auth = auth_agg[auth_agg['user_id'].isin(u_train)].copy()
    dev_auth   = auth_agg[auth_agg['user_id'].isin(u_dev)].copy()
    test_auth  = auth_agg[auth_agg['user_id'].isin(u_test)].copy()

    # 5) Sample authentic train
    pos_train  = train_auth[train_auth['label'] == 'positive']
    ctrl_train = train_auth[train_auth['label'] == 'control']
    n_pos      = int(len(pos_train) * AUTH_POS_PCT)
    n_ctrl     = int(len(ctrl_train) * AUTH_CTRL_PCT)
    pos_sample = pos_train.sample(n=n_pos, random_state=42)
    ctrl_sample= ctrl_train.sample(n=n_ctrl, random_state=42)
    train_auth_samp = pd.concat([pos_sample, ctrl_sample], ignore_index=True)

    # 6) Sample synthetic augment
    syn_pos_ids     = agg_syn_pos['user_id'].unique()
    syn_ctrl_ids    = agg_syn_ctrl['user_id'].unique()
    n_syn_pos_keep  = int(len(syn_pos_ids) * AUG_SYN_POS_PCT)
    n_syn_ctrl_keep = int(len(syn_ctrl_ids) * AUG_SYN_CTRL_PCT)
    np.random.seed(42)
    syn_pos_keep    = np.random.choice(syn_pos_ids, n_syn_pos_keep, replace=False)
    syn_ctrl_keep   = np.random.choice(syn_ctrl_ids, n_syn_ctrl_keep, replace=False)
    train_syn_pos   = agg_syn_pos[agg_syn_pos['user_id'].isin(syn_pos_keep)].copy()
    train_syn_ctrl  = agg_syn_ctrl[agg_syn_ctrl['user_id'].isin(syn_ctrl_keep)].copy()
    train_syn_pos['y']  = 1
    train_syn_ctrl['y'] = 0

    # 7) Build feature matrices
    train_combined = pd.concat([train_auth_samp, train_syn_pos, train_syn_ctrl], ignore_index=True)
    X_tr = train_combined[feat_cols].values
    y_tr = train_combined['y'].values
    assert not np.isnan(y_tr).any(), "y_tr contains NaNs!"

    X_dev = dev_auth[feat_cols].values
    y_dev = dev_auth['y'].values
    X_te  = test_auth[feat_cols].values
    y_te  = test_auth['y'].values

    # 8) Grid-search on train/dev
    best_score, best_params = -1, None
    for params in ParameterGrid(PARAM_GRID):
        clf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        clf.fit(X_tr, y_tr)
        f1 = f1_score(y_dev, clf.predict(X_dev))
        if f1 > best_score:
            best_score, best_params = f1, params
    print(f"Best params (dev F1={best_score:.3f}): {best_params}")

    # 9) Final train on train+dev
    final_train = pd.concat([train_auth_samp, dev_auth, train_syn_pos, train_syn_ctrl], ignore_index=True)
    X_final     = final_train[feat_cols].values
    y_final     = final_train['y'].values
    final_clf   = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    final_clf.fit(X_final, y_final)

    # 10) Evaluate on test
    y_te_pred = final_clf.predict(X_te)
    y_te_prob = final_clf.predict_proba(X_te)[:, 1]
    print("\n=== Test set classification (user-level) ===")
    print(classification_report(y_te, y_te_pred, target_names=['control', 'positive']))
    print("Test AUC:", roc_auc_score(y_te, y_te_prob))

    # 11) UMAP & distributional dists
    plot_umap(auth_agg[feat_cols], auth_agg['label'], os.path.join(FIG_DIR, 'umap_authentic.png'))
    ctrl_agg = auth_agg[auth_agg['label'] == 'control']
    pos_agg  = auth_agg[auth_agg['label'] == 'positive']
    df_dist  = compute_and_plot_dists(ctrl_agg, pos_agg, feat_cols)

    # 12) Top divergent features
    print("\nTop 10 JS divergence features:\n", df_dist['JS_divergence'].nlargest(10))
    print("\nTop 10 Wasserstein features:\n", df_dist['Wasserstein'].nlargest(10))
