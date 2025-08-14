import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths to your files – update these to the correct locations
CSV_PATH    = 'IMPORTANT files/GoogleDriveData/Combined_balanced_bipolar.csv'
JSONL_PATH  = 'IMPORTANT files/GoogleDriveData/bipolar.metadata.jsonl'
OUTPUT_PATH = 'IMPORTANT files/GoogleDriveData/filtered_merged_POSTS_30-200w_0-200p.csv'

# Filter parameters – adjust as needed
MIN_WORDS = 30      # minimum words per post
MAX_WORDS = 200     # maximum words per post
MIN_POSTS = 0       # minimum posts per user
MAX_POSTS = 200     # maximum posts per user

# 1. Load datasets
df_csv   = pd.read_csv(CSV_PATH, dtype={'user_id': str, 'post_id': str})
df_jsonl = pd.read_json(JSONL_PATH, lines=True, dtype={'user_id': str, 'post_id': str})

# 2. Merge on user_id and post_id
df = pd.merge(df_csv, df_jsonl, on=['user_id', 'post_id'], how='inner')

# 2a. Keep only top‐level posts (remove comments)
# df = df[df['type'] == 'post'].copy()

# 3. Filter posts by word count
mask_wc = df['word_count'].between(MIN_WORDS, MAX_WORDS)
df_filtered = df[mask_wc].copy()

# 4. Compute per-user post counts on the filtered set
user_counts = df_filtered.groupby('user_id').size()

# 5. Identify users within the posts range
good_users = user_counts[(user_counts >= MIN_POSTS) & (user_counts <= MAX_POSTS)].index

# 6. Keep only posts by valid users
df_final = df_filtered[df_filtered['user_id'].isin(good_users)].copy()

# 7. Write to CSV
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"Exported {len(df_final)} rows to {OUTPUT_PATH}")

# 8. Generate 3D distribution graph
#    - X: posts per user
#    - Y: avg. word count per post
#    - Z: number of users in each bin

# Compute metrics
user_post_counts = df_final.groupby('user_id').size()
user_word_sums   = df_final.groupby('user_id')['word_count'].sum()
user_avg_words   = user_word_sums / user_post_counts

# Define bin ranges
x_bins = np.arange(0, MAX_POSTS+1, 5)
y_bins = np.arange(0, MAX_WORDS+1, 5)

# 2D histogram
hist, x_edges, y_edges = np.histogram2d(
    user_post_counts,
    user_avg_words,
    bins=[x_bins, y_bins]
)

# Prepare 3D bar coordinates
dx = x_bins[1] - x_bins[0]
dy = y_bins[1] - y_bins[0]
_x, _y = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing='ij')
x = _x.ravel()
y = _y.ravel()
z = np.zeros_like(x)
dz = hist.ravel()

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x, y, z, dx, dy, dz, shade=True)
ax.set_xlabel('Posts per User')
ax.set_ylabel('Avg. Word Count per Post')
ax.set_zlabel('Number of Users')
plt.title('3D Distribution: Posts × Avg Words × Users')
plt.show()
