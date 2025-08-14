#!/usr/bin/env python3
import pandas as pd
import sys

# Hardcoded file paths (update these to your actual file locations)
INPUT_CSV_PATH = 'IMPORTANT files/FinalDatasets/TraitDebiasing/MAZS_male_posts_2700.csv'
OUTPUT_CSV_PATH = 'IMPORTANT files/FinalDatasets/TraitDebiasing/formatted_MAZS_male_posts_2700.csv'

def convert_first_to_second(input_csv: str, output_csv: str):
    """
    Reads a “first‐format” CSV (must contain a column named “Generated Text"),
    and writes a “second‐format” CSV with columns:
      TID, user_id, post_id, label, text, language

    - user_id: integers 1..N (one per row)
    - post_id: set to 1 for all rows
    - TID: “<user_id>_<post_id>”
    - label: “bipolar"
    - text: from "Generated Text" column
    - language: "en"
    """
    # 1) Read the input CSV
    try:
        df_in = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error: could not read '{input_csv}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Verify "Generated Text" column exists
    if "generated_text" not in df_in.columns:
        print(
            "Error: input CSV must contain a column named 'Generated Text'.",
            file=sys.stderr
        )
        print(f"Available columns: {df_in.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # 3) Build new columns
    N = len(df_in)
    user_ids = list(range(1, N + 1))
    post_ids = [1] * N
    tids = [f"{uid}_{pid}" for uid, pid in zip(user_ids, post_ids)]
    labels = ["bipolar"] * N
    texts = df_in["generated_text"].astype(str).tolist()
    languages = ["en"] * N

    # 4) Create the output DataFrame
    df_out = pd.DataFrame({
        "TID": tids,
        "user_id": user_ids,
        "post_id": post_ids,
        "label": labels,
        "text": texts,
        "language": languages,
    })

    # 5) Write to output CSV
    try:
        df_out.to_csv(output_csv, index=False)
        print(f"Successfully wrote {N} rows to '{output_csv}'")
    except Exception as e:
        print(f"Error: could not write '{output_csv}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    convert_first_to_second(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
