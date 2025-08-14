import os
import pandas as pd

# === USER INPUT ===
INPUT_CSV  = 'files/FinalDatasets/clean_New800_META_1000Attribute_Zero_Shot.csv'        # Replace with your actual input path
OUTPUT_CSV = 'files/FinalDatasets/FIRST454_clean_New800_META_1000Attribute_Zero_Shot.csv'   # Replace with your desired output path
N_ROWS     = 454

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Read the first N_ROWS from the input CSV
df_head = pd.read_csv(INPUT_CSV, nrows=N_ROWS)

# Write them out to the new CSV
df_head.to_csv(OUTPUT_CSV, index=False)

print(f"Saved first {N_ROWS} rows from '{INPUT_CSV}' to '{OUTPUT_CSV}'")
