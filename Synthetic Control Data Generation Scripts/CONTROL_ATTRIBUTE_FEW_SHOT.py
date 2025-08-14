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
import random

# === Helper: Sample and format few-shot examples ===
def get_numbered_texts(df, num_posts=5, seed=42):
    sampled_texts = df['text'].dropna().sample(n=num_posts, random_state=seed).tolist()
    return "\n".join([f"{i+1}. {text}" for i, text in enumerate(sampled_texts)])


def main():
    try:
        print("Starting few-shot generation pipeline...")

        # Load few-shot datasets
        bipolar_df = pd.read_csv("/home/yc663354/testing/GenerationScripts/POSTS_bipolar_unique_100.csv")
        control_df = pd.read_csv("/home/yc663354/testing/GenerationScripts/POSTS_control_unique_100.csv")

        # Load external profile data
        occupations_data = load_occupations()
        interests_data = load_interests()
        subreddits_data = load_subreddits()
        countries_data = load_nationalities()

        # Model setup
        model_path = "/hpcwork/thes2019/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=35, verbose=False)

        # Settings
        total_posts = 5400
        few_shot_examples = 5
        condition = "Bipolar Disorder"

        # Build few-shot context
        control_context = get_numbered_texts(control_df, few_shot_examples)
        few_shot_block = (
            "Here are example comments written by control users:\n\n"
            f"{control_context}\n\n"
        )

        data_prompt_template = """
Now independently imagine yourself as a normal reddit user and write exactly one reddit comment that would fit in the subreddit {sub_choice}.
Generate only one comment per iteration.
Your comment must be approximately 40-60 words long. Do not use any hashtags in your comment. Do not use greetings like "hello" or exclamations like "wow"
at the start. Exclude preambles. Be creative with your response. Focus on general behavior, expression, and tone. Do not use a title. 
You should imitate the examples I have provided, but you cannot simply modify or rewrite the examples I have given.
"""

        diversity_prompt = """
Provide something more diverse than the previous posts.
Change the structure at the beginning of your response: it shouldn't follow the format of your previous posts.
"""

        gen_config = {
            "max_tokens": 2000,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": ["</s>"]
        }

        prompts = []
        for i in range(1, total_posts + 1):
            profile = generate_profile(
                occupations=occupations_data,
                interests=interests_data,
                subreddits=subreddits_data,
                countries=countries_data
            )
            profile["condition"] = condition
            sub_choice = profile["subreddit"]

            data_prompt = data_prompt_template.format(sub_choice=sub_choice)
            if i % 5 == 0:
                body = f"{data_prompt}\n\n{diversity_prompt}"
                ptype = "diversity"
            else:
                body = data_prompt
                ptype = "normal"

            persona_line = (
                f"Your age is: {profile['age']}. You are a {profile['gender']} from {profile['nationality']}. "
                f"Your occupation is: {profile['occupation']}. Your marital status is: {profile['marital_status']}. "
                f"Your interests are: {', '.join(profile['interests'])}. "
            )

            full_prompt = f"[INST] {persona_line}\n\n{few_shot_block}{body} [/INST]"

            prompts.append({
                "index": i,
                "type": ptype,
                "prompt": full_prompt,
                "persona_line": persona_line,
                "profile": profile
            })

        results = []
        start = datetime.now()

        for idx, entry in enumerate(tqdm(prompts, desc="Generating comments"), start=1):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    out = llm(entry["prompt"], **gen_config)
                text = out["choices"][0]["text"].strip()
                # Remove surrounding quotes if present
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
            except Exception as e:
                text = f"[Error: {e}]"

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
                "Marital Status": profile["marital_status"],
                "Condition": profile["condition"],
                "Persona": entry["persona_line"],
                "Prompt": entry["prompt"],
                "Generated Text": text,
                **gen_config,
                "stop": str(gen_config["stop"]),
                "model": os.path.basename(model_path)
            })

        end = datetime.now()
        print(f"\nDone in {end - start}")

        df = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfn = f"/home/yc663354/testing/GenerationScripts/CONTROLScripts/FINALVERSION_CONTROL_5400_Attribute_Few_Shot.csv"
        df.to_csv(outfn, index=False)
        print(f"Saved {len(df)} rows to {outfn}")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
