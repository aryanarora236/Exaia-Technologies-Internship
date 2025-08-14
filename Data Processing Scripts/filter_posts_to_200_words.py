import pandas as pd

def filter_posts_to_200_words(df, text_col='text', user_col='user_id', max_words=1500):
    """
    Filters posts from each user until the total word count reaches max_words.
    """
    filtered_rows = []

    # Group by user_id
    for user, group in df.groupby(user_col):
        total_words = 0

        # Iterate through posts in order
        for _, row in group.iterrows():
            post_text = str(row[text_col])
            post_words = len(post_text.split())

            if total_words + post_words > max_words:
                break

            filtered_rows.append(row)
            total_words += post_words

    return pd.DataFrame(filtered_rows)

# === USAGE EXAMPLE ===

# Load CSV
input_path = "files/GoogleDriveData/balanced_control_part1.csv"  # <- change this to your file path
df = pd.read_csv(input_path)

# Filter posts
filtered_df = filter_posts_to_200_words(df)

# Save result
output_path = "files/Input_INUSEFiles/CONTROL_100Users_max1500words.csv"
filtered_df.to_csv(output_path, index=False)

print(f"Filtered data saved to: {output_path}")
