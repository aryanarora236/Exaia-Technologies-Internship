#!/usr/bin/env python3
import os
import pandas as pd
import io
import contextlib
from datetime import datetime
from tqdm import tqdm
from llama_cpp import Llama

from profile_generator import (
    generate_profile,
    load_occupations,
    load_interests,
    load_subreddits,
    load_nationalities
)

def clean_comment(text):
    # strip stray quotes/spaces
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()
    return text

def main():
    # --------------------------------
    # 1) Load your input personas CSV
    # --------------------------------
    input_csv = "Final5400MistralPersonas.csv"
    print(f"Loading personas from {input_csv}…")
    df_input = pd.read_csv(input_csv, dtype=str)

    # --------------------------------
    # 2) Load profile‐generation data
    # --------------------------------
    occupations_data = load_occupations()
    interests_data = load_interests()
    subreddits_data = load_subreddits()
    countries_data = load_nationalities()

    # --------------------------------
    # 3) Initialize Meta‐Llama‐3
    # --------------------------------
    model_path = "/hpcwork/thes2019/models/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf"
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_gpu_layers=82,
        verbose=True
    )

    # --------------------------------
    # 4) Generation hyper‐params
    # --------------------------------
    gen_config = {
        "max_tokens": 270,
        "temperature": 0.95,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.10,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }

    # --------------------------------
    # 5) Banned‐starts/contains/ends & diversity
    # --------------------------------
    banned_starts = (
        "you are", "as a", "hello", "hi", "i am", "as someone", "this is my comment",
        "my comment", "dear", "greetings", "i'm", "as an", "i have", "i think",
        "in my opinion", "i've noticed", "recently", "it’s interesting",
        "for me", "it’s funny", "in my experience", "personally", "i always",
        "i've been", "i've always", "i find", "one thing", "just wanted to say",
        "to me", "i wonder"
    )
    banned_contains = (
        "bipolar", "mental health", "diagnosed", "condition",
        "as a person", "as an ai", "as an assistant"
    )
    banned_ends = (
        "hope this helps", "thanks for reading", "that's my comment",
        "that's it", "just my thoughts", "anyone else", "anyone else?",
        "anyone else.", "anyone else have", "does anyone else", "anyone have",
        "what do you think", "has anyone", "who else", "do you agree?",
        "is it just me?", "am i the only one?"
    )

    diversity_instruction = (
        "Make your writing style or structure different from previous comments. "
        "Try a new approach or tone."
    )

    # --------------------------------
    # 6) Build prompts, one per row
    # --------------------------------
    prompts = []
    for idx, row in df_input.iterrows():
        # generate a random profile (for subreddit choice + metadata)
        profile = generate_profile(
            occupations=occupations_data,
            interests=interests_data,
            subreddits=subreddits_data,
            countries=countries_data
        )

        sub_choice = profile["subreddit"]
        persona_line = "You are " + row["model_response"].strip()

        # core instructions from snippet #2, but replace the attribute‐based persona
        prompt_body = (
            f"Write a single, realistic Reddit comment that is about 70–80 words for r/{sub_choice}. "
            f"{persona_line} "
            "You are diagnosed with bipolar disorder. (do not mention the diagnosis or mental health in your comment)"
            "Write only the comment—no greetings, sign‐offs, summaries, or meta-statements. "
            "Do not start your comment with phrases like 'Just', 'Here', 'As a', 'I am', 'This is', "
            "'I'm', greetings, or phrases like 'I've been', 'I spent', 'I just', 'I recently', or similar. "
            "Start directly with a statement, opinion, detail, or description—avoid Reddit formulaic openers. "
            "Try not to use the same opener or sentence structure as earlier comments. Use unique phrasings. "
            "Sound casual and authentic, like a real Redditor. "
            "If the persona is married, do not always mention or focus on your spouse—write as you would on Reddit. "
            "Do not end your comment by asking 'anyone else?' or similar."
        )
        # diversity tweak every 10th
        if (idx + 1) % 10 == 0:
            prompt_body += " " + diversity_instruction

        full_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant that writes realistic, Reddit-style comments based on persona and subreddit context.\n"
            "Your outputs should strictly follow the instructions.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt_body}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        prompts.append({
            "user_id": row.get("user_id", ""),
            "post_id": row.get("post_id", ""),
            "prompt": full_prompt,
            "profile": profile,
            "persona_line": persona_line
        })

    # --------------------------------
    # 7) Run inference + post‐process
    # --------------------------------
    results = []
    start = datetime.now()
    print("Generating comments with retry/clean logic…")
    for entry in tqdm(prompts, desc="Meta‐Llama inference"):
        comment = ""
        # up to 6 tries
        for _ in range(6):
            try:
                out = llm(entry["prompt"], **gen_config)
                text = out["choices"][0].get("text", "").strip()
                text = clean_comment(text)
                bad = (
                    not text
                    or text.lower().startswith(banned_starts)
                    or any(b in text.lower() for b in banned_contains)
                    or any(text.lower().endswith(e) for e in banned_ends)
                    or len(text.split()) < 20
                )
                if not bad:
                    comment = text
                    break
            except Exception:
                # on exception, loop to retry
                continue
        if not comment:
            comment = "[No valid comment generated]"

        p = entry["profile"]
        results.append({
            "user_id": entry["user_id"],
            "post_id": entry["post_id"],
            "Age": p["age"],
            "Gender": p["gender"],
            "Education": p["education"],
            "Occupation": p["occupation"],
            "Interests": ", ".join(p["interests"]),
            "Subreddit": p["subreddit"],
            "Nationality": p["nationality"],
            "Marital Status": p["marital_status"],
            # if your CSV persona already encodes condition, profile.get(...) will default
            "Condition": p.get("condition", "Bipolar Disorder"),
            "Persona": entry["persona_line"],
            "Prompt": entry["prompt"],
            "Generated Text": comment
        })

    end = datetime.now()
    print(f"Done in {end - start}")

    # --------------------------------
    # 8) Dump to CSV
    # --------------------------------
    df_out = pd.DataFrame(results)
    outfn = f"FINALVERSION_5400_MetaLlama_InferredZeroShot.csv"
    df_out.to_csv(outfn, index=False)
    print(f"Saved {len(df_out)} rows to {outfn}")

if __name__ == "__main__":
    main()
