import sys
import pandas as pd

# ─── CONFIGURE THESE ──────────────────────────────────────────────────────────
INPUT_PATH  = "IMPORTANT files/FinalDatasets/PROCESSING/synthetic/FINALVERSION_5400_META_Attribute_Zero_Shot.csv"       # ← your TSV/CSV
OUTPUT_PATH = "IMPORTANT files/FinalDatasets/TraitDebiasing/MAZS_male_posts_2700.csv" # ← desired output path
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Attempt to read with automatic delimiter detection
    try:
        df = pd.read_csv(
            INPUT_PATH,
            sep=None,
            engine="python",
            dtype=str,
            keep_default_na=False
        )
    except Exception as e:
        sys.exit(f"Error reading '{INPUT_PATH}': {e}")

    # Normalize column names to simple lowercase keys
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )

    # Ensure we have a gender column
    if "gender" not in df.columns:
        print(f"ERROR: No 'gender' column found in '{INPUT_PATH}'.")
        print("Available columns:")
        for col in df.columns:
            print(f" - {col}")
        sys.exit(1)

    # Filter for male
    male_df = df[df["gender"].str.strip().str.lower() == "male"]
    count = len(male_df)

    if count < 2700:
        print(f"Found only {count} male posts in '{INPUT_PATH}'; not enough to extract 2700.")
        sys.exit(1)

    # Take exactly the first 2700
    selected = male_df.iloc[:2700]

    # Write out
    try:
        selected.to_csv(OUTPUT_PATH, index=False)
        print(f"Wrote {len(selected)} male posts from '{INPUT_PATH}' to '{OUTPUT_PATH}'.")
    except Exception as e:
        sys.exit(f"Error writing '{OUTPUT_PATH}': {e}")

if __name__ == "__main__":
    main()
