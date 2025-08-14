import pandas as pd
import random

def sample_unique_user_posts(input_csv, output_csv, num_posts=100):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Ensure there are enough unique users
    unique_users = df['user_id'].unique()
    if len(unique_users) < num_posts:
        raise ValueError(f"Not enough unique users to sample {num_posts} posts.")

    # Randomly select the user IDs
    selected_users = random.sample(list(unique_users), num_posts)

    # Get one random post per selected user
    sampled_df = (
        df[df['user_id'].isin(selected_users)]
        .groupby('user_id')
        .apply(lambda group: group.sample(1))
        .reset_index(drop=True)
    )

    # Save to a new CSV file
    sampled_df.to_csv(output_csv, index=False)
    print(f"Saved {num_posts} unique posts to {output_csv}")

# Example usage
sample_unique_user_posts(
    input_csv='files/FinalDatasets/AuthenticBipolarPosts.csv',
    output_csv='FEWSHOT_SAMPLES/POSTS_bipolar_unique_100.csv',
    num_posts=100
)
