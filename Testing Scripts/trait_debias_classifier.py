#!/usr/bin/env python3
"""
cymo_user_rf_augment.py

1. Load sentence-level CYMO CSVs:
     - authentic positive
     - authentic control
     - synthetic positive
     - synthetic control
2. Extract user_id from TID.
3. Keep only the first N unique users per dataset.
4. Aggregate (mean) features per user.
5. Build single train/dev split from all authentic users **excluding** manual lists:
     - Train/dev pool = authentic users not in manual lists.
     - Test set = all users from both manual_demographics.csv (positives) and control_manual_demographics.csv (controls).
6. Augment **training** with fractions of authentic positive, authentic control, synthetic positive, and synthetic control.
7. Grid‐search RF on train/dev (maximize user‐level F1).
8. Train final RF on train+dev (augmented), evaluate on combined manual-test set.
9. UMAP + distributional analyses on aggregated authentic user features.
"""

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
AUTH_POS_PATH       = 'IMPORTANT files/CYMO_Outputs/AUTHENTIC_USING_DATASETS/Combined_CYMO_bipolar.csv'
AUTH_CTRL_PATH      = 'IMPORTANT files/CYMO_Outputs/AUTHENTIC_USING_DATASETS/ann.balanced_control_part1.csv'
SYN_POS_PATH        = 'IMPORTANT files/FinalDatasets/TraitDebiasing/ann.formatted_mazs_male_posts_2700.csv'
SYN_CTRL_PATH       = 'IMPORTANT files/CYMO_Outputs/FINAL/SYNTHETIC CONTROL/ann.formatted_finalversion_5400_control_meta_attribute_zero_shot.csv'
MANUAL_PATH         = 'IMPORTANT files/testing/FINALFILES/Metadata T_MHC 100 posts - Sheet1 (8).csv'             # authentic positive users
CONTROL_MANUAL_PATH = 'IMPORTANT files/FinalDatasets/CONTROL Metadata T_MHC 100 posts - Sheet1.csv'     # authentic control users
FIG_DIR             = 'AnalysisResults/REAL_FINAL/figures_user_augment_MAZS_withgender_1.0synthetic/'
os.makedirs(FIG_DIR, exist_ok=True)

TOTAL_USERS       = 5400
AUTH_POS_PCT      = 1.0
AUTH_CTRL_PCT     = 1.0
AUG_SYN_POS_PCT   = 1.0
AUG_SYN_CTRL_PCT  = 1.0

PARAM_GRID = {
    'n_estimators':      [50, 100],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2]
}

NON_FEATURE_COLS = ['TID', 'sid', 'label', 'source', 'model']

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def extract_user_id(df):
    df = df.copy()
    df['user_id'] = df['TID'].astype(str).str.split('_').str[0]
    return df


def limit_users(df, max_users):
    uids = df['user_id'].unique()[:max_users]
    return df[df['user_id'].isin(uids)].copy()


def aggregate_user_features(df):
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + ['TID','sid','user_id','label']]
    agg = df.groupby('user_id')[feat_cols].mean().reset_index()
    return agg, feat_cols


def plot_umap(features, labels, fname):
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(features)
    dfu = pd.DataFrame(emb, columns=['UMAP1','UMAP2'])
    dfu['label'] = labels.values
    plt.figure(figsize=(8,6))
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
        plt.figure(figsize=(6,4))
        sns.kdeplot(a, label='control', fill=True, alpha=0.5)
        sns.kdeplot(p, label='positive', fill=True, alpha=0.5)
        plt.title(f"{feat} — JS={js:.3f}, WD={wd:.3f}")
        plt.xlabel(feat); plt.legend()
        plt.savefig(os.path.join(FIG_DIR, f"density_{feat}.png"), dpi=150)
        plt.close()
        results.append({'feature': feat, 'JS_divergence': js, 'Wasserstein': wd})
    dfres = pd.DataFrame(results).set_index('feature')
    dfres.to_csv(os.path.join(FIG_DIR,'feature_distance_summary.csv'))
    return dfres

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load & aggregate datasets
    df_pos_auth  = pd.read_csv(AUTH_POS_PATH).pipe(extract_user_id).pipe(limit_users, TOTAL_USERS); df_pos_auth['label']='positive'
    df_ctrl_auth = pd.read_csv(AUTH_CTRL_PATH).pipe(extract_user_id).pipe(limit_users, TOTAL_USERS); df_ctrl_auth['label']='control'
    df_syn_pos   = pd.read_csv(SYN_POS_PATH).pipe(extract_user_id).pipe(limit_users, TOTAL_USERS); df_syn_pos['label']='positive'
    df_syn_ctrl  = pd.read_csv(SYN_CTRL_PATH).pipe(extract_user_id).pipe(limit_users, TOTAL_USERS); df_syn_ctrl['label']='control'
    agg_pos_auth, feat_cols = aggregate_user_features(df_pos_auth); agg_pos_auth['label']='positive'
    agg_ctrl_auth, _        = aggregate_user_features(df_ctrl_auth); agg_ctrl_auth['label']='control'
    agg_syn_pos, _          = aggregate_user_features(df_syn_pos); agg_syn_pos['label']='positive'
    agg_syn_ctrl, _         = aggregate_user_features(df_syn_ctrl); agg_syn_ctrl['label']='control'

    # All authentic user-level features
    auth_agg = pd.concat([agg_ctrl_auth, agg_pos_auth], ignore_index=True)
    auth_agg['y'] = auth_agg['label'].map({'control':0,'positive':1})

    # Manual test set IDs
    manual_df  = pd.read_csv(MANUAL_PATH, dtype={'user_id':str})
    control_df = pd.read_csv(CONTROL_MANUAL_PATH, dtype={'user_id':str})
    test_ids = set(manual_df['user_id']) | set(control_df['user_id'])

    # Create test set = only manual users
    test_df = auth_agg[auth_agg['user_id'].isin(test_ids)].copy()

    # Pool for train/dev = all others
    pool_df = auth_agg[~auth_agg['user_id'].isin(test_ids)].copy()
    u_pool, y_pool = pool_df['user_id'].values, pool_df['y'].values
    dev_frac = 0.1/0.9
    u_train, u_dev, _, _ = train_test_split(u_pool, y_pool, test_size=dev_frac, stratify=y_pool, random_state=42)
    train_df = pool_df[pool_df['user_id'].isin(u_train)].copy()
    dev_df   = pool_df[pool_df['user_id'].isin(u_dev)].copy()

    # Pre-sample synthetic augment
    syn_pos_ids  = agg_syn_pos['user_id'].unique()
    syn_ctrl_ids = agg_syn_ctrl['user_id'].unique()
    np.random.seed(42)
    syn_pos_keep  = np.random.choice(syn_pos_ids, int(len(syn_pos_ids)*AUG_SYN_POS_PCT), replace=False)
    syn_ctrl_keep = np.random.choice(syn_ctrl_ids, int(len(syn_ctrl_ids)*AUG_SYN_CTRL_PCT), replace=False)
    train_syn_pos  = agg_syn_pos[agg_syn_pos['user_id'].isin(syn_pos_keep)].copy(); train_syn_pos['y']=1
    train_syn_ctrl = agg_syn_ctrl[agg_syn_ctrl['user_id'].isin(syn_ctrl_keep)].copy(); train_syn_ctrl['y']=0

    # Pipeline runner
    def run_pipeline(train_df, dev_df, test_df, scenario_name='ManualTest'):
        # Sample auth train
        pos_train  = train_df[train_df['label']=='positive']
        ctrl_train = train_df[train_df['label']=='control']
        pos_samp   = pos_train.sample(frac=AUTH_POS_PCT, random_state=42)
        ctrl_samp  = ctrl_train.sample(frac=AUTH_CTRL_PCT, random_state=42)
        train_samp = pd.concat([pos_samp, ctrl_samp], ignore_index=True)

        # Build X/y
        df_tr = pd.concat([train_samp, train_syn_pos, train_syn_ctrl], ignore_index=True)
        X_tr, y_tr = df_tr[feat_cols].values, df_tr['y'].values
        X_dev, y_dev = dev_df[feat_cols].values, dev_df['y'].values
        X_te, y_te   = test_df[feat_cols].values, test_df['y'].values

        # Grid search
        best_score, best_params = -1, None
        for p in ParameterGrid(PARAM_GRID):
            clf = RandomForestClassifier(random_state=42, n_jobs=-1, **p)
            clf.fit(X_tr, y_tr)
            f1 = f1_score(y_dev, clf.predict(X_dev))
            if f1 > best_score:
                best_score, best_params = f1, p
        print(f"[{scenario_name}] Best dev F1={best_score:.3f}, params={best_params}")

        # Final train + eval
        final_df = pd.concat([train_samp, dev_df, train_syn_pos, train_syn_ctrl], ignore_index=True)
        clf_final = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
        clf_final.fit(final_df[feat_cols].values, final_df['y'].values)

        y_pred = clf_final.predict(X_te)
        unique = np.unique(y_te)
        print(f"\n[{scenario_name}] Test classification:")
        if len(unique) < 2:
            print(f"Only one class present ({unique}). Skipping report & AUC.")
        else:
            print(classification_report(y_te, y_pred, labels=[0,1], target_names=['control','positive']))
            y_prob = clf_final.predict_proba(X_te)[:,1]
            print(f"[{scenario_name}] Test AUC: {roc_auc_score(y_te, y_prob)}")
        print()

    # Run once
    run_pipeline(train_df, dev_df, test_df)

    # UMAP + dist analyses on all authentic
    plot_umap(auth_agg[feat_cols], auth_agg['label'], os.path.join(FIG_DIR,'umap_authentic.png'))
    ctrl_all = auth_agg[auth_agg['label']=='control']
    pos_all  = auth_agg[auth_agg['label']=='positive']
    df_dist  = compute_and_plot_dists(ctrl_all, pos_all, feat_cols)
    print("Top 10 JS divergence features:\n", df_dist['JS_divergence'].nlargest(10))
    print("Top 10 Wasserstein features:\n", df_dist['Wasserstein'].nlargest(10))
