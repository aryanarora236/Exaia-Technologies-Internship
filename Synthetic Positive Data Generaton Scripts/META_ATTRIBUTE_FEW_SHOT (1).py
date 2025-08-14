#!/usr/bin/env python3
import os
import io
import contextlib
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

# === Helper: Clean and format ===
def clean_comment(text):
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text.strip()

# === Helper: Few-shot examples sampler ===
def get_numbered_texts(df, num_posts=3, seed=42):
    sampled = df['text'].dropna().sample(n=num_posts, random_state=seed).tolist()
    return "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(sampled)])


def main():
    try:
        print("Starting few-shot generation pipeline...")

        # Load few-shot example datasets
        bipolar_df = pd.read_csv("POSTS_bipolar_unique_100.csv")
        control_df = pd.read_csv("POSTS_control_unique_100.csv")
        few_shot_examples = 3
        bipolar_context = get_numbered_texts(bipolar_df, few_shot_examples)
        control_context = get_numbered_texts(control_df, few_shot_examples)
        few_shot_block = (
            "Here are some example comments for reference:\n\n"
            f"Examples from users with bipolar disorder:\n{bipolar_context}\n\n"
            f"Examples from control users:\n{control_context}\n\n"
        )

        # Load external profile data
        occupations_data = load_occupations()
        interests_data = load_interests()
        subreddits_data = load_subreddits()
        countries_data = load_nationalities()

        # Model setup
        model_path = "/hpcwork/thes2019/models/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf"
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=82,
            verbose=True
        )

        total_posts = 5400
        gen_config = {
            "max_tokens": 3000,
            "temperature": 0.95,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.10,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }

        # Banned opener/contains/closer lists
        banned_starts = (
            "you are", "as a", "hello", "hi", "i am", "as someone", "this is my comment",
            "my comment", "dear", "greetings", "i'm", "as an", "i have", "i think", "in my opinion",
            "i've noticed", "recently", "it’s interesting", "for me", "it’s funny", "in my experience",
            "personally", "i always", "i've been", "i've always", "i find", "one thing", "just wanted to say",
            "to me", "i wonder"
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
            sub_choice = profile['subreddit']
            interests = ', '.join(profile['interests'])
            condition = profile.get('condition', 'Bipolar Disorder')

            persona = (
                f"You are a {profile['age']}-year-old {profile['gender']} from "
                f"{profile['nationality']}, {profile['marital_status']}, working as a "
                f"{profile['occupation']}, interested in {interests}, diagnosed with "
                f"{condition} (do not mention the diagnosis or mental health in your comment)."
            )

            # Build the prompt body with few-shot block + instructions
            prompt_body = (
                f"Write a single, realistic Reddit comment that is about 70-80 words for r/{sub_choice}. "
                f"{persona} Write only the comment—no greetings, sign-offs, summaries, or meta-statements. "
                f"Do not say 'Here is a reddit comment' or anything similar to that in the beginning of the output. "
                f"Do not start your comment with phrases like 'Just', 'Here', 'As a', 'I am', 'This is', "
                f"'I'm', greetings, or phrases like 'I've been', 'I spent', 'I just', 'I recently', or similar. "
                f"Start directly with a statement, opinion, detail, or description—avoid Reddit formulaic openers. "
                f"Try not to use the same opener or sentence structure as earlier comments. Use unique phrasings. "
                f"Sound casual and authentic, like a real Redditor. "
                f"If the persona is married, do not always mention or focus on your spouse in your comment—"
                f"write as you would on Reddit, where many users do not reference their marriage directly in every post. "
                f"Do not end your comment by asking the crowd for agreement or saying 'anyone else?' or similar phrases. "
                f"Vary your language and sentence structure from comment to comment. "
                f"{few_shot_block}"
            )

            if i % 5 == 0:
                prompt_body += f" {diversity_instruction}"

            # Append the imitation clause and wrap in INST token
            full_prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are a helpful assistant that writes realistic, Reddit-style comments based on persona and subreddit context.\n"
                "Your outputs should strictly follow the instructions.\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{prompt_body}\n"
                f"You should imitate the examples I have provided, but you cannot simply modify or rewrite "
                f"the examples I have given./n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )

            prompts.append({
                'index': i,
                'prompt': full_prompt
            })

        # Generation loop
        results = []
        start = datetime.now()
        for entry in tqdm(prompts, desc="Generating comments"):
            idx = entry['index']
            comment = ''
            for _ in range(4):
                out = llm(entry['prompt'], **gen_config)
                text = clean_comment(out.get('choices', [{}])[0].get('text', ''))
                if (
                    text
                    and not text.lower().startswith(banned_starts)
                    and not any(bad in text.lower() for bad in banned_contains)
                    and not any(text.lower().endswith(end) for end in banned_ends)
                    and len(text.split()) >= 20
                ):
                    comment = text
                    break
            results.append({'TID': idx, 'Generated Text': comment or "[No valid comment]"})

            # reset KV-cache periodically
            if idx % 600 == 0:
                llm.reset()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

        end = datetime.now()
        print(f"\nDone in {end - start}")

        df = pd.DataFrame(results)
        outfn = f"/home/yc663354/testing/GenerationScripts/FINALVERSION_5400_META_ATTRIBUTE_FEW_SHOT.csv"
        df.to_csv(outfn, index=False)
        print(f"Saved {len(df)} rows to {outfn}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == '__main__':
    main()
