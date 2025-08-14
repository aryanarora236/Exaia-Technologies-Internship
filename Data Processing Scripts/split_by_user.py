#!/usr/bin/env python3
"""
split_by_user.py

Reads a large CSV and writes out roughly the first 100 000 rows,
but never splits a given userID across files. In other words,
once you hit 100 000 rows, you keep writing any additional rows
belonging to that same userID, then stop.

Assumes that each row’s TID column is “<userID>_<something>” (e.g. “847043_wkxrmo”),
so it infers userID = the substring before the first underscore.
"""

import csv
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_CSV = "balanced_control_part1.csv"     # path to your large CSV
OUTPUT_CSV = "CONTROL_first_100k_unprocessed.csv"   # where to write the truncated output
ROW_LIMIT = 100_000             # target row count before finishing a user
# ──────────────────────────────────────────────────────────────────────────────

def split_keep_whole_users(input_path: str, output_path: str, limit: int):
    """
    Reads input_path line-by-line, writes to output_path.
    Once we've written `limit` rows, we record the userID of that last row
    and keep writing only rows that have the same userID. As soon as a different
    userID appears, we stop.
    """
    with open(input_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        # Read header and write it immediately
        header = next(reader)
        writer.writerow(header)
        
        total_written = 0
        last_user = None
        done_finalizing = False
        
        # Find index of the TID column (in case it's not the first)
        try:
            tid_idx = header.index("TID")
        except ValueError:
            sys.exit("ERROR: No 'TID' column found in header.")
        
        for row in reader:
            # Extract userID = substring before the first underscore in TID
            full_tid = row[tid_idx]
            if "_" not in full_tid:
                # If format is unexpected, just take the entire TID as userID
                current_user = full_tid
            else:
                current_user = full_tid.split("_", 1)[0]
            
            if total_written < limit:
                # Still under the limit, write unconditionally
                writer.writerow(row)
                total_written += 1
                last_user = current_user
            
            else:
                # We have reached (or surpassed) the limit; now only write
                # rows from the same last_user. As soon as we see a different
                # user, break out.
                if not done_finalizing:
                    if current_user == last_user:
                        writer.writerow(row)
                        # total_written += 1      # you can increment if you like,
                        # but we don't strictly need this once limit is reached
                    else:
                        # different userID => we’re done
                        done_finalizing = True
                        break
                else:
                    break
        
        print(f"Done: wrote {total_written} rows (plus any extra rows for user '{last_user}').")

if __name__ == "__main__":
    split_keep_whole_users(INPUT_CSV, OUTPUT_CSV, ROW_LIMIT)
