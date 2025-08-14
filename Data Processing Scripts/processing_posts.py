#!/usr/bin/env python3
import pandas as pd
import sys

# ─── Configure your paths here ───────────────────────────────────────
INPUT_PATH    = "files/GoogleDriveData/CONTROL_matched_data.csv"
OUTPUT_PATH   = "files/GoogleDriveData/CONTROL_processed_data.csv"
SEGMENT_SIZE  = 100  # words per segment
# ────────────────────────────────────────────────────────────────────

def split_text_into_segments(text, segment_size):
    """Split text into consecutive segments of up to segment_size words."""
    words = text.split()
    return [' '.join(words[i:i+segment_size]) 
            for i in range(0, len(words), segment_size)]

def process_all_posts():
    # 1. Load
    try:
        df = pd.read_csv(INPUT_PATH, dtype={'user_id': str, 'post_id': str})
    except Exception as e:
        sys.exit(f"Error reading {INPUT_PATH}: {e}")

    # 2. Keep only posts
    df = df[df['type'] == 'post'].copy()

    # 3. Sort each user's posts by created_utc
    df.sort_values(['user_id', 'created_utc'], inplace=True)

    # 4. Split long posts
    output_rows = []
    for _, row in df.iterrows():
        segments = split_text_into_segments(row['text'], SEGMENT_SIZE)
        for idx, seg in enumerate(segments, start=1):
            new = row.copy()
            new['text']        = seg
            new['word_count']  = len(seg.split())
            new['TID']         = f"{row['TID']}_{idx}"
            output_rows.append(new)

    processed = pd.DataFrame(output_rows)

    # 5. Write out
    try:
        processed.to_csv(OUTPUT_PATH, index=False)
        print(f"Output {len(processed)} rows → {OUTPUT_PATH}")
    except Exception as e:
        sys.exit(f"Error writing {OUTPUT_PATH}: {e}")

if __name__ == "__main__":
    process_all_posts()
