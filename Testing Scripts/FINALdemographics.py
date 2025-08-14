#!/usr/bin/env python3

import re
import json
import pandas as pd
from collections import defaultdict

# === Normalization helpers with fixed bins ===

def bin_age(age_str):
    """
    Normalize age strings into bins: <18, 18-25, 26-35, 36-50, 50+ or unknown.
    """
    age_str = age_str.lower().strip()
    # Attempt numeric extraction
    try:
        age = int(re.sub(r"[^\d]", "", age_str))
        if age < 18:
            return "<18"
        elif age <= 25:
            return "18-25"
        elif age <= 35:
            return "26-35"
        elif age <= 50:
            return "36-50"
        else:
            return "50+"
    except Exception:
        pass

    # Implicit phrases
    implicit_bins = {
        "preteen": "<18", "minor": "<18", "teenager": "<18", "teen": "<18",
        "young adult": "18-25", "twentysomething": "18-25", "early 20s": "18-25",
        "mid 20s": "18-25", "late 20s": "18-25", "college": "18-25", "university": "18-25",
        "high school": "<18", "middle school": "<18", "elementary school": "<18",
        "kid": "<18", "child": "<18", "too young to drive": "<18",
        "too young to drink": "<18", "too young to vote": "<18",
        "finally legal": "18-25", "legal age": "18-25", "approaching 18": "<18",
        "almost 18": "<18",
    }
    for phrase, bin_label in implicit_bins.items():
        if phrase in age_str:
            return bin_label
    return "unknown"


def normalize_gender(g):
    """
    Normalize gender-related strings into Female, Male, Non-binary/other, or Unknown.
    """
    text = re.sub(r'[^a-z\s]', '', str(g).lower().strip())
    tokens = set(text.split())

    female = {"female", "girl", "woman", "lady", "f"}
    male = {"male", "man", "boy", "g"}
    nonb = {"nonbinary", "nb", "they", "other"}
    trans = {"trans", "transgender", "mtf", "ftm"}

    if not text or text in {"na", "none", "unknown", "unspecified", "prefer not to say", "nan"}:
        return "Unknown"
    if female & tokens:
        return "Female"
    if male & tokens:
        return "Male"
    if trans & tokens or nonb & tokens:
        return "Non-binary/other"

    # Fallback substring
    for term in female:
        if term in text:
            return "Female"
    for term in male:
        if term in text:
            return "Male"
    for term in trans | nonb:
        if term in text:
            return "Non-binary/other"
    return "Unknown"


def normalize_education(edu):
    """
    Normalize education strings into High School, Community College, Bachelor's, Master's, or Unknown.
    """
    text = " " + re.sub(r'[^a-z0-9\s]', '', str(edu).lower().strip()) + " "
    grad = {"phd", "doctorate", "doctoral", "dr", "dphil", "master", "masters", "msc", "ms", "ma", "mba"}
    bach = {"bachelor", "ba", "bs", "bsc"}
    comm = {"community college", "assoc degree", "associates", "aa", "as", "some college"}
    hs = {"high school", "secondary", "hs", "ged"}

    for terms, label in [
        (grad, "Master’s Degree"),
        (bach, "Bachelor’s Degree"),
        (comm, "Community College"),
        (hs, "High School")
    ]:
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", text):
                return label
    return "Unknown"

# === Pattern definitions (explicit & implicit) ===
EXPLICIT_PATTERNS = {
    "age": [
        (re.compile(r"\bI[' ]?m (\d{1,2})\b", re.IGNORECASE), "I'm {age}"),
        (re.compile(r"\bI am (\d{1,2})\b", re.IGNORECASE), "I am {age}"),
        (re.compile(r"\bI was (\d{1,2})\b", re.IGNORECASE), "I was {age}"),
        (re.compile(r"\b(\d{1,2})\s?[mMfF]\b"), "{age}M/F"),
        (re.compile(r"\bI[' ]?m a (\d{1,2})[- ]?(year[- ]old|yo)\b", re.IGNORECASE), "I'm a {age} year old"),
        (re.compile(r"\bjust turned (\d{1,2})\b", re.IGNORECASE), "Just turned {age}"),
        (re.compile(r"\bturned (\d{1,2}) (today|last week|this week)\b", re.IGNORECASE), "Turned {age} recently"),
        (re.compile(r"\bwhen I was (?:about|around)? ?(\d{1,2})\b", re.IGNORECASE), "When I was {age}"),
        (re.compile(r"\bin (\d{1,2})(?:st|nd|rd|th)? grade\b", re.IGNORECASE), "In {age}th grade"),
        (re.compile(r"\bstarted high school at (\d{1,2})\b", re.IGNORECASE), "Started high school at {age}"),
        (re.compile(r"\bage[:：]?\s*(\d{1,2})\b", re.IGNORECASE), "age: {age}"),
        (re.compile(r"\bjust reached (\d{1,2})\b", re.IGNORECASE), "Just reached {age}"),
        (re.compile(r"\b(?:only|still) (\d{1,2})\b", re.IGNORECASE), "Still/only {age}"),
        (re.compile(r"\bnow that I'm (\d{1,2})\b", re.IGNORECASE), "Now that I'm {age}"),
        (re.compile(r"\bI'm now (\d{1,2})\b", re.IGNORECASE), "I'm now {age}"),
        (re.compile(r"\bI'm already (\d{1,2})\b", re.IGNORECASE), "I'm already {age}"),
    ],
    "gender": [
        (re.compile(r"\bI[' ]?m (?:a[n]? )?(guy|boy|man|male|female|woman|girl)\b", re.IGNORECASE), "I am a {gender}"),
        (re.compile(r"\bI[' ]?m a (?:cis|trans)? ?(guy|girl|man|woman)\b", re.IGNORECASE), "I am a {gender}"),
        (re.compile(r"\bmy pronouns are (he/him|she/her|they/them|xe/xem|ze/zir|any pronouns|it/its)\b", re.IGNORECASE), "Pronouns: {gender}"),
        (re.compile(r"\bi[' ]?m transgender\b", re.IGNORECASE), "Transgender"),
        (re.compile(r"\bi[' ]?m non[- ]?binary\b", re.IGNORECASE), "Nonbinary"),
        (re.compile(r"\bi[' ]?m (cisgender|cis|trans|nonbinary|genderqueer|genderfluid|agender)\b", re.IGNORECASE), "Gender identity: {gender}"),
        (re.compile(r"\bas a (guy|man|girl|woman)\b", re.IGNORECASE), "As a {gender}"),
        (re.compile(r"\bi[' ]?ve always been a (boy|girl|guy|woman|man)\b", re.IGNORECASE), "Always been {gender}"),
    ],
    "education": [
        (re.compile(r"\b(I[' ]?m|currently)?\s*(studying|study|studied|majoring in)\s+(.*?)(college|university|school)\b", re.IGNORECASE), "Studying at {edu}"),
        (re.compile(r"\bI[' ]?m (?:a[n]? )?(.*?)(student|major|undergrad|grad)\b", re.IGNORECASE), "{edu} student"),
        (re.compile(r"\bjust got accepted to (college|university|med school|law school)\b", re.IGNORECASE), "Accepted to {edu}"),
        (re.compile(r"\benrolled in (an?|a)? ?(engineering|design|psych|cs|medical|nursing|humanities|business|arts) program\b", re.IGNORECASE), "Program: {edu}"),
        (re.compile(r"\b(freshman|sophomore|junior|senior) (in|at)? ?(college|university|high school)?\b", re.IGNORECASE), "{edu} year"),
        (re.compile(r"\bgraduated with (?:a|an)? ?(BA|BS|MA|MS|PhD|associate's|bachelor's|master's) in\b", re.IGNORECASE), "Holds {edu}"),
    ]
}

IMPLICIT_PATTERNS = {
    "age": [
        (re.compile(r"\bi[' ]?m (?:a )?teenager\b", re.IGNORECASE), "Teenager"),
        (re.compile(r"\bi[' ]?m a preteen\b", re.IGNORECASE), "Preteen"),
        (re.compile(r"\bi[' ]?m a minor\b", re.IGNORECASE), "Minor"),
        (re.compile(r"\bi[' ]?m under\s+(\d{2})\b", re.IGNORECASE), "Under {age}"),
        (re.compile(r"\byoung adult\b", re.IGNORECASE), "Young adult"),
        (re.compile(r"\btwentysomething\b", re.IGNORECASE), "Twentysomething"),
        (re.compile(r"\bi[' ]?m too young to (drive|drink|vote)\b", re.IGNORECASE), "Too young to {activity}"),
    ],
    "gender": [
        (re.compile(r"\bi [' ]?m attracted to (men|women|both|everyone)\b", re.IGNORECASE), "Attracted to {gender}"),
        (re.compile(r"\bmy (boyfriend|girlfriend)\b", re.IGNORECASE), "Partner implies gender"),
        (re.compile(r"\b(he|she|they) said I was (beautiful|handsome|cute)\b", re.IGNORECASE), "Compliment implies gender perception"),
    ],
    "education": [
        (re.compile(r"\btaking a gap year\b", re.IGNORECASE), "Gap year"),
        (re.compile(r"\bdorm\b|\bcampus\b|\bstudent center\b|\blibrary\b", re.IGNORECASE), "Student life location"),
        (re.compile(r"\balumni|alumnus|graduate\b", re.IGNORECASE), "Graduate or former student"),
    ]
}

# === Extraction helpers ===

def extract_all_demographics(text):
    matches = []
    for trait in ["age", "gender", "education"]:
        for pattern, _ in EXPLICIT_PATTERNS.get(trait, []):
            for match in pattern.finditer(text):
                matches.append({
                    "trait": trait,
                    "disclosure_type": "explicit",
                    "matched_phrase": match.group(0)
                })
        for pattern, _ in IMPLICIT_PATTERNS.get(trait, []):
            for match in pattern.finditer(text):
                matches.append({
                    "trait": trait,
                    "disclosure_type": "implicit",
                    "matched_phrase": match.group(0)
                })
    return matches


def sliding_windows(text, window_size=100, overlap=50):
    words = text.split()
    windows = []
    start = 0
    while start < len(words):
        end = start + window_size
        windows.append(" ".join(words[start:end]))
        start += (window_size - overlap)
    return windows

# === Manual processing ===

def process_manual_comments_demographics(csv_path, jsonl_path):
    df = pd.read_csv(csv_path, dtype={"user_id": str})
    user_ids = set(df["user_id"].dropna())

    all_results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            uid = str(item.get("user_id", ""))
            if uid not in user_ids:
                continue

            text = item.get("text", "")
            created = item.get("created_utc", 0)

            matches = extract_all_demographics(text)
            for m in matches:
                norm = m["matched_phrase"]
                if m["trait"] == "age":
                    norm = bin_age(norm)
                elif m["trait"] == "gender":
                    norm = normalize_gender(norm)
                elif m["trait"] == "education":
                    norm = normalize_education(norm)

                all_results.append({
                    "UserID": f"user_{uid}",
                    "Trait": m["trait"],
                    "DisclosureType": m["disclosure_type"],
                    "MatchedPhrase": m["matched_phrase"],
                    "Normalized": norm,
                    "CreatedUTC": created,
                    "OriginalText": (text[:200] + "...") if len(text) > 200 else text
                })

    if not all_results:
        return []

    df_res = pd.DataFrame(all_results)
    df_res = df_res.sort_values("CreatedUTC", ascending=False)
    df_dedup = df_res.drop_duplicates(subset=["UserID", "Trait"], keep="first")
    return df_dedup

# === Automated processing ===

def process_automated_user_chunks(csv_path, jsonl_path):
    df = pd.read_csv(csv_path, dtype={"user_id": str})
    user_ids = set(df["user_id"].dropna())
    user_chunks = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            uid = str(item.get("user_id", ""))
            if uid not in user_ids:
                continue

            user_chunks[uid].append(item.get("text", ""))

    all_results = []
    for uid, chunks in user_chunks.items():
        combined = " ".join(chunks)
        windows = sliding_windows(combined, 50, 25)
        for i, window in enumerate(windows, 1):
            matches = extract_all_demographics(window)
            for m in matches:
                norm = m["matched_phrase"]
                if m["trait"] == "age":
                    norm = bin_age(norm)
                elif m["trait"] == "gender":
                    norm = normalize_gender(norm)
                elif m["trait"] == "education":
                    norm = normalize_education(norm)

                all_results.append({
                    "UserID": f"user_{uid}",
                    "Trait": m["trait"],
                    "DisclosureType": m["disclosure_type"],
                    "MatchedPhrase": m["matched_phrase"],
                    "Normalized": norm,
                    "ChunkID": i
                })

    if not all_results:
        return pd.DataFrame([])

    df_res = pd.DataFrame(all_results)
    df_dedup = df_res.drop_duplicates(subset=["UserID", "Trait"], keep="first")
    return df_dedup

# === Distribution computation ===

def compute_distributions(df, output_path=None):
    dist = df.groupby(["Trait", "Normalized"]).size().reset_index(name="Count")
    if output_path:
        dist.to_csv(output_path, index=False)
    print(dist)
    return dist

# === Main script ===

if __name__ == "__main__":
    csv_path = "IMPORTANT files/testing/FINALFILES/Metadata T_MHC 100 posts - Sheet1 (8).csv"
    jsonl_path = "IMPORTANT files/GoogleDriveData/everythingmatched.jsonl"

    # Manual results
    manual_df = process_manual_comments_demographics(csv_path, jsonl_path)
    manual_df.to_csv("manual_results.csv", index=False)
    compute_distributions(manual_df, "manual_distribution.csv")

    # Automated results
    automated_df = process_automated_user_chunks(csv_path, jsonl_path)
    automated_df.to_csv("automated_results.csv", index=False)
    compute_distributions(automated_df, "automated_distribution.csv")

    print(f"[✓] Saved manual ({len(manual_df)}) & automated ({len(automated_df)}) records and distributions.")
