#!/usr/bin/env python3
import os
import pandas as pd
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
    # Remove leading/trailing quotes and stray spaces
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.strip()
    # Remove leading/trailing single quotes too, if present
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text.strip()

def main():
    try:
        print("Starting main generation pipeline...")

        occupations_data = load_occupations()
        interests_data = load_interests()
        subreddits_data = load_subreddits()
        countries_data = load_nationalities()

        model_path = "/hpcwork/thes2019/models/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf"
        llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_gpu_layers=82,
            verbose=True
        )

        total_posts = 5400

        gen_config = {
            "max_tokens": 270,
            "temperature": 0.95,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.10,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }

        # Expanded to catch more formulaic openings
        banned_starts = (
            "you are", "as a", "hello", "hi", "i am", "as someone", "this is my comment",
            "my comment", "dear", "greetings", "i'm", "as an", "i have", "i think", "in my opinion",
            "i've noticed", "recently", "it’s interesting", "for me", "it’s funny", "in my experience",
            "personally", "i always", "i've been", "i've always", "i find", "one thing", "just wanted to say", "to me", "i wonder"
        )
        banned_contains = (
            "bipolar", "mental health", "diagnosed", "condition", "as a person", "as an ai", "as an assistant"
        )
        banned_ends = (
            "hope this helps", "thanks for reading", "that's my comment", "that's it", "just my thoughts",
            "anyone else", "anyone else?", "anyone else.", "anyone else have", "does anyone else", "anyone have",
            "what do you think", "has anyone", "who else", "do you agree?", "is it just me?", "am i the only one?"
        )

        diversity_instruction = (
            "Make your writing style or structure different from previous comments. Try a new approach or tone."
        )

        prompts = []
        for i in range(1, total_posts + 1):
            profile = generate_profile(
                occupations=occupations_data,
                interests=interests_data,
                subreddits=subreddits_data,
                countries=countries_data
            )
            sub_choice = profile["subreddit"]
            interests = ', '.join(profile["interests"])
            condition = profile.get("condition", "Bipolar Disorder")

            persona = (
                f"You are a {profile['age']}-year-old {profile['gender']} from {profile['nationality']}, "
                f"{profile['marital_status']}, working as a {profile['occupation']}, "
                f"interested in {interests}, diagnosed with {condition} (do not mention the diagnosis or mental health in your comment)."
            )

            # Data prompt for realism and variety
            prompt_body = (
                f"Write a single, realistic Reddit comment that is about 70-80 words for r/{sub_choice}. "
                f"{persona} Write only the comment—no greetings, sign-offs, summaries, or meta-statements. "
                f"Do not start your comment with phrases like 'Just', 'Here', 'As a', 'I am', 'This is', 'I'm', greetings, or phrases like                     'I've been', 'I spent', 'I just', 'I recently', or similar. "
                f"Start directly with a statement, opinion, detail, or description—avoid Reddit formulaic openers. "
                f"Try not to use the same opener or sentence structure as earlier comments. Use unique phrasings."
                f"Sound casual and authentic, like a real Redditor. "
                f"If the persona is married, do not always mention or focus on your spouse in your comment—write as you would on Reddit, where                 many users do not reference their marriage directly in every post."
                f"Do not end your comment by asking the crowd for agreement or saying 'anyone else?' or similar phrases. "
                f"Vary your language and sentence structure. Generate exactly one comment."
            )

            if i % 10 == 0:
                prompt_body = f"{prompt_body} {diversity_instruction}"

            full_prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are a helpful assistant that writes realistic, Reddit-style comments based on persona and subreddit context.\n"
                "Your outputs should strictly follow the instructions.\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{prompt_body}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                 )

            prompts.append({
                "index": i,
                "prompt": full_prompt,
                "profile": profile,
                "persona_line": persona
            })

        results = []
        start = datetime.now()
        for idx, entry in enumerate(tqdm(prompts, desc="Generating comments"), start=1):
            comment = ""
            for attempt in range(6):  # try up to 2 times for a valid response
                try:
                    out = llm(
                        entry["prompt"],
                        max_tokens=gen_config["max_tokens"],
                        temperature=gen_config["temperature"],
                        top_p=gen_config["top_p"],
                        top_k=gen_config["top_k"],
                        repeat_penalty=gen_config["repeat_penalty"],
                        presence_penalty=gen_config["presence_penalty"],
                        frequency_penalty=gen_config["frequency_penalty"]
                    )
                    text = out["choices"][0].get("text", "").strip() if ("choices" in out and out["choices"]) else ""
                    text = clean_comment(text)
                    bad = (
                        not text or
                        text.lower().startswith(banned_starts) or
                        any(badword in text.lower() for badword in banned_contains) or
                        any(text.lower().endswith(end) for end in banned_ends) or
                        len(text.split()) < 20
                    )
                    if not bad:
                        comment = text
                        break
                except Exception as e:
                    text = f"[Error: {e}]"
            if not comment:
                comment = "[No valid comment generated]"

            profile = entry["profile"]
            results.append({
                "TID": idx,
                "Age": profile["age"],
                "Gender": profile["gender"],
                "Education": profile["education"],
                "Occupation": profile["occupation"],
                "Interests": ', '.join(profile["interests"]),
                "Subreddit": profile["subreddit"],
                "Nationality": profile["nationality"],
                "Marrital Status": profile["marital_status"],
                "Condition": condition,
                "Persona": entry["persona_line"],
                "Prompt": entry["prompt"],
                "Generated Text": comment,
            })

        end = datetime.now()
        print(f"\nDone in {end - start}")

        df = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfn = f"/home/yc663354/testing/GenerationScripts/FINALVERSION_5400_Attribute_Zero_Shot.csv"
        df.to_csv(outfn, index=False)
        print(f"Saved {len(df)} rows to {outfn}")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
