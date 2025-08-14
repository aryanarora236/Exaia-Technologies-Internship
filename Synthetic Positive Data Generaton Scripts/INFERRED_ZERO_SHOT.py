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

def main():
    # --------------------------------
    # Ask user for the CSV path
    # --------------------------------
    input_csv = "Final5400MistralPersonas.csv"
    try:
        print(f"Loading input CSV from {input_csv}...")
        df_input = pd.read_csv(input_csv, dtype=str)

        # ================================
        # Load External Profile Data
        # ================================
        occupations_data = load_occupations()
        interests_data = load_interests()
        subreddits_data = load_subreddits()
        countries_data = load_nationalities()

        # ================================
        # Model Initialization
        # ================================
        model_path = "/hpcwork/thes2019/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=35,
            verbose=False
        )

        # ================================
        # Generation Settings
        # ================================
        condition = "Bipolar Disorder"
        data_prompt_template = """
You are diagnosed with Bipolar Disorder.
Now independently imagine yourself as a normal reddit user and write exactly one reddit comment that would fit in the subreddit {sub_choice}.
Your comment must be approximately 77 words long. Do not use any hashtags in your comment. Do not use greetings like "hello" or exclamations like "wow"
at the start. Exclude preambles. The mention of any explicit reference to bipolar disorder is not allowed. 
You must avoid using any language related to mental health such as a condition or general terms like "mental illness", 
"diagnosed with", or "suffering from". Be creative with your response. Focus on general behavior, expression, and tone. Do not use a title.
"""
        diversity_prompt = """
Provide something more diverse than the previous posts.
Change the structure at the beginning of your response: it shouldn't follow the format of your previous posts.
"""
        gen_config = {
            "max_tokens":       360,
            "temperature":      0.8,
            "top_p":            0.9,
            "top_k":            20,
            "repeat_penalty":   1.1,
            "presence_penalty": 0.0,
            "frequency_penalty":0.0,
            "stop":             ["</s>"]
        }

        # ================================
        # Build prompts
        # ================================
        prompts = []
        total = len(df_input)
        print(f"Preparing {total} prompts based on input CSV…")

        for idx, row in df_input.iterrows():
            # generate a random profile as before
            profile = generate_profile(
                    occupations=occupations_data,
                    interests=interests_data,
                    subreddits=subreddits_data,
                    countries=countries_data
                )
            profile["condition"] = condition

            # prepend "You are " to the persona from the CSV
            persona_line = "You are " + row["model_response"].strip()
            sub_choice   = profile["subreddit"]
            data_prompt  = data_prompt_template.format(sub_choice=sub_choice)

            # every 5th prompt, add diversity instruction
            if (idx + 1) % 5 == 0:
                body = f"{data_prompt}\n\n{diversity_prompt}"
                ptype = "diversity"
            else:
                body = data_prompt
                ptype = "normal"

            full_prompt = f"[INST] {persona_line}\n\n{body} [/INST]"

            prompts.append({
                "user_id":      row["user_id"],
                "post_id":      row["post_id"],
                "type":         ptype,
                "prompt":       full_prompt,
                "persona_line": persona_line,
                "profile":      profile
            })

        # ================================
        # Run Model Inference
        # ================================
        results = []
        start = datetime.now()
        print("Generating comments…")
        for entry in tqdm(prompts, desc="LLM inference"):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    out = llm(
                        entry["prompt"],
                        max_tokens=gen_config["max_tokens"],
                        temperature=gen_config["temperature"],
                        top_p=gen_config["top_p"],
                        top_k=gen_config["top_k"],
                        repeat_penalty=gen_config["repeat_penalty"],
                        presence_penalty=gen_config["presence_penalty"],
                        frequency_penalty=gen_config["frequency_penalty"],
                        stop=gen_config["stop"]
                    )
                generated = out["choices"][0]["text"].strip()
            except Exception as e:
                generated = f"[Error: {e}]"

            profile = entry["profile"]
            results.append({
                "user_id":        entry["user_id"],
                "post_id":        entry["post_id"],
                "Age":            profile["age"],
                "Gender":         profile["gender"],
                "Education":      profile["education"],
                "Occupation":     profile["occupation"],
                "Interests":      ', '.join(profile["interests"]),
                "Subreddit":      profile["subreddit"],
                "Nationality":    profile["nationality"],
                "Marital Status": profile["marital_status"],
                "Condition":      profile["condition"],
                "Persona":        entry["persona_line"],
                "Prompt":         entry["prompt"],
                "Generated Text": generated,
                **gen_config,
                "model":          os.path.basename(model_path)
            })

        end = datetime.now()
        print(f"\nDone in {end - start}")

        # ================================
        # Save Output to CSV
        # ================================
        df_out = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfn = f"/home/yc663354/testing/GenerationScripts/FINALVERSION_5400_Inferred_Zero_Shot.csv"
        df_out.to_csv(outfn, index=False)
        print(f"Saved {len(df_out)} rows to {outfn}")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
