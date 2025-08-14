#!/usr/bin/env python3
import pandas as pd
import sys

# ─── Configure your paths here ───────────────────────────────────────
CSV_PATH    = "IMPORTANT files/GoogleDriveData/Combined_balanced_bipolar.csv"
JSONL_PATH  = "IMPORTANT files/GoogleDriveData/bipolar.metadata.jsonl"
OUTPUT_PATH = "IMPORTANT files/GoogleDriveData/everythingmatched.jsonl"
# ────────────────────────────────────────────────────────────────────

def load_csv(path):
    """Load the CSV of posts."""
    try:
        return pd.read_csv(path, dtype={'user_id': str, 'post_id': str})
    except Exception as e:
        sys.exit(f"Error reading CSV file {path}: {e}")


def load_jsonl(path):
    """Load the JSONL metadata."""
    try:
        return pd.read_json(path, lines=True, dtype={'user_id': str, 'post_id': str})
    except Exception as e:
        sys.exit(f"Error reading JSONL file {path}: {e}")


def merge_data(df_posts, df_meta):
    """Inner-join on user_id & post_id."""
    return pd.merge(
        df_posts,
        df_meta,
        on=['user_id', 'post_id'],
        how='inner'
    )


def main():
    # Load posts CSV
    df_posts = load_csv(CSV_PATH)
    # Load metadata JSONL
    df_meta  = load_jsonl(JSONL_PATH)

    # Merge posts with their metadata
    df_merged = merge_data(df_posts, df_meta)

    # Write output JSONL
    try:
        df_merged.to_json(
            OUTPUT_PATH,
            orient='records',
            lines=True,
            force_ascii=False
        )
        print(f"Merged {len(df_merged)} rows → {OUTPUT_PATH}")
    except Exception as e:
        sys.exit(f"Error writing merged JSONL: {e}")

if __name__ == "__main__":
    main()
