#!/usr/bin/env python3
import os
import pandas as pd

# ← EDIT THIS to point at your CSV:
input_path = 'IMPORTANT files/FinalDatasets/PROCESSING/control/FINALVERSION_5400_CONTROL_META_Inferred_Few_Shot.csv'

# Build output filename by appending "_cleaned" before the extension
base, ext = os.path.splitext(input_path)
output_path = f"IMPORTANT files/FinalDatasets/CLEANED/clean_FINALVERSION_5400_CONTROL_META_Inferred_Few_Shot.csv"

# --- processing ---
# Read the CSV
df = pd.read_csv(input_path)

# Ensure the 'Generated Text' column is present
if 'Generated Text' not in df.columns:
    raise KeyError("Column 'Generated Text' not found in the input CSV.")

# Filter out rows with:
#  • "[No valid comment generated]" anywhere in the text
#  • Generated Text starting with "Here"
#  • Generated Text starting with "AI"
mask = ~(
    df['Generated Text'].astype(str).str.contains(r"\[No valid comment generated\]") |
    df['Generated Text'].astype(str).str.startswith("Here") |
    df['Generated Text'].astype(str).str.startswith("AI")
)
clean_df = df[mask]

# Save the cleaned DataFrame
clean_df.to_csv(output_path, index=False)

# Print summary
removed = len(df) - len(clean_df)
print(f"Removed {removed} rows; cleaned file saved to:\n  {output_path}")
print(f"Rows in cleaned file: {len(clean_df)}")
