import pandas as pd

def get_median_length_post_per_user(df, text_col='text', user_col='user_id'):
    """
    For each user, return only the post with the median length.
    """
    median_posts = []

    for user, group in df.groupby(user_col):
        # Add a word count column
        group = group.copy()
        group['word_count'] = group[text_col].apply(lambda x: len(str(x).split()))

        # Sort by word count
        group_sorted = group.sort_values(by='word_count').reset_index(drop=True)
        median_index = len(group_sorted) // 2  # Lower median if even number of posts

        # Select the median-length post
        median_post = group_sorted.iloc[median_index]
        median_posts.append(median_post)

    return pd.DataFrame(median_posts)

# === USAGE EXAMPLE ===

# Load CSV
input_path = "files/GoogleDriveData/balanced_bipolar_part3.csv" 
df = pd.read_csv(input_path)

# Get median-length post per user
median_posts_df = get_median_length_post_per_user(df)

# Drop the extra 'word_count' column before saving
median_posts_df = median_posts_df.drop(columns=['word_count'])

# Save to CSV
output_path = "Part3_BIPOLAR_MHC_median_length_posts_per_user.csv"
median_posts_df.to_csv(output_path, index=False)

print(df['user_id'].nunique(), len(median_posts_df))
print(f"Median-length posts saved to: {output_path}")
