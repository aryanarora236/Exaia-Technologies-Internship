import pandas as pd

# Configuration: specify file paths and parameters here
INPUT_CSV = "files/Input_INUSEFiles/T_MHC_100Users_max1500words.csv"
OUTPUT_CSV = "first50_posts.csv"
USER_COL = "user_id"  # Column name identifying each user
N_USERS = 50           # Number of unique users to select

def filter_first_n_users(input_csv: str, output_csv: str, user_col: str, n_users: int):
    """
    Reads a CSV file, selects the first n unique users from the specified user column,
    and writes all rows (posts) belonging to those users to a new CSV file.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path where the filtered CSV will be saved.
        user_col (str): Name of the column identifying users.
        n_users (int): Number of unique users to select.
    """
    # Load the data
    df = pd.read_csv(input_csv)

    # Get the first n unique users in order of appearance
    unique_users = df[user_col].dropna().unique()[:n_users]

    # Filter DataFrame for these users
    filtered_df = df[df[user_col].isin(unique_users)]

    # Save to output CSV
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data for first {n_users} users saved to '{output_csv}'")

if __name__ == '__main__':
    filter_first_n_users(
        INPUT_CSV,
        OUTPUT_CSV,
        USER_COL,
        N_USERS
    )
