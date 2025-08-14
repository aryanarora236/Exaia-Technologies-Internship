import pandas as pd
import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from llama_cpp import Llama
from datetime import datetime

# Import profile generator utilities
from profile_generator import (
    generate_profile,
    load_occupations,
    load_interests,
    load_subreddits,
    load_nationalities
)

class PersonaFewShotGenerator:
    def __init__(
        self,
        embeddings_file: str = '/home/yc663354/testing/data_files/part1control_embedded.csv',
        personas_file: str = '/home/yc663354/testing/GenerationScripts/CONTROLScripts/Final5400MistralPersonas.csv',
        metadata_file: str = '/home/yc663354/testing/GenerationScripts/NEWControlMinWord30MaxWord150.csv',
        max_personas: int = None
    ):
        # Normalize file paths
        self.embeddings_file = os.path.abspath(os.path.expanduser(embeddings_file))
        self.personas_file = os.path.abspath(os.path.expanduser(personas_file))
        self.metadata_file = os.path.abspath(os.path.expanduser(metadata_file))
        self.max_personas = max_personas

        # Embedding model setup
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.num_clusters = 200
        self.embeddings_df = None
        self.centroids = None
        self.clusters = None

        # Load metadata for TID filter
        print(f"Loading metadata from {self.metadata_file}...")
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        meta_df = pd.read_csv(self.metadata_file)
        if 'TID' not in meta_df.columns:
            raise ValueError("Metadata file must contain a 'TID' column")
        self.valid_tids = set(meta_df['TID'].astype(str))

        # Load profile generation data
        print("Loading profile data...")
        self.occupations_data = load_occupations()
        self.interests_data = load_interests()
        self.subreddits_data = load_subreddits()
        self.countries_data = load_nationalities()

        # Initialize Meta-Llama-3
        model_path = os.path.expanduser(
            '/hpcwork/thes2019/models/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf'
        )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=82,
            verbose=True
        )

        # Generation config
        self.gen_config = {
            "max_tokens": 2048,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }

        # Banned patterns & diversity instruction
        self.banned_starts = (
            "you are", "as a", "hello", "hi", "i am", "as someone", "this is my comment",
            "my comment", "dear", "greetings", "as an",
            "in my opinion", "recently", "for me", "it’s funny", "in my experience", "personally", "i always",
            "i've been", "one thing", "just wanted to say",
            "to me", "i wonder"
        )
        self.banned_contains = (
            "bipolar", "mental health", "diagnosed", "condition",
            "as a person", "as an ai", "as an assistant"
        )
        self.banned_ends = (
            "hope this helps", "thanks for reading", "that's my comment",
            "that's it", "just my thoughts", "anyone else", "anyone else?",
            "anyone else.", "anyone else have", "does anyone else", "anyone have",
            "what do you think", "has anyone", "who else", "do you agree?",
            "is it just me?", "am i the only one?"
        )
        self.diversity_instruction = (
            "Make your writing style or structure different from previous comments. "
            "Try a new approach or tone."
        )

        # Cache reset interval
        self.reset_interval = 400  # reset KV-cache after this many generations

    def load_embeddings(self) -> None:
        import ast
        print(f"Loading embeddings from {self.embeddings_file}...")
        df = pd.read_csv(self.embeddings_file, dtype=str)
        if 'TID' not in df.columns:
            raise ValueError("Embeddings file must contain a 'TID' column")
        df = df[df['TID'].isin(self.valid_tids)].reset_index(drop=True)
        print(f"Filtered embeddings to {len(df)} rows based on metadata TIDs.")
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        embeddings = np.vstack(df['embedding'].values)

        if 'comment_text' not in df.columns:
            df['comment_text'] = df.get(
                'text', pd.Series([f"Comment {i}" for i in range(len(df))])
            )

        self.embeddings_df = df
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=1000,
            random_state=42,
            n_init=3
        )
        kmeans.fit(embeddings)
        self.centroids = kmeans.cluster_centers_
        self.clusters = kmeans.labels_
        self.embeddings_df['cluster'] = self.clusters

    def select_examples(self, emb: np.ndarray) -> list:
        """
        Pick top, mid, and bottom clusters, then return both most- and least-similar comment from each.
        """
        sims = cosine_similarity(emb.reshape(1, -1), self.centroids)[0]
        cid_top = np.argmax(sims)
        cid_mid = np.argsort(sims)[len(sims) // 2]
        cid_bot = np.argmin(sims)
        examples = []
        for cid in (cid_top, cid_mid, cid_bot):
            sub = self.embeddings_df[self.embeddings_df['cluster'] == cid].copy()
            arr = np.vstack(sub['embedding'].tolist())
            sims_sub = cosine_similarity(emb.reshape(1, -1), arr)[0]
            sub['sim'] = sims_sub
            sub_sorted = sub.sort_values('sim', ascending=False)
            most = sub_sorted.iloc[0]['comment_text']
            least = sub_sorted.iloc[-1]['comment_text']
            examples.extend([most, least])
        return examples

    def clean_comment(self, text: str) -> str:
        text = text.strip()
        if (
            (text.startswith('"') and text.endswith('"')) or
            (text.startswith("'") and text.endswith("'"))
        ):
            text = text[1:-1].strip()
        return text

    def process_all_personas(self) -> pd.DataFrame:
        print(f"Loading personas from {self.personas_file}...")
        df_personas = pd.read_csv(self.personas_file, dtype=str)
        df_personas.columns = df_personas.columns.str.strip()
        key = (
            'model_response' if 'model_response' in df_personas.columns else
            'persona' if 'persona' in df_personas.columns else
            'original_text'
        )
        df_personas['persona_text'] = df_personas[key]

        if self.max_personas:
            df_personas = df_personas.head(self.max_personas)

        self.load_embeddings()

        results = []
        print("Generating comments with Meta-Llama-3...")
        for idx, row in tqdm(df_personas.iterrows(), total=len(df_personas)):
            persona_line = "You are " + row['persona_text'].strip()
            profile = generate_profile(
                occupations=self.occupations_data,
                interests=self.interests_data,
                subreddits=self.subreddits_data,
                countries=self.countries_data
            )
            sub_choice = profile['subreddit']

            emb = self.model.encode([row['persona_text']])[0]
            examples = self.select_examples(emb)

            few_shot_block = "Here are example comments for reference:\n"
            for i, ex in enumerate(examples, 1):
                few_shot_block += f"{i}. {ex}\n"

            prompt_body = (
                f"Write a single, realistic Reddit comment that is about 40–60 words for r/{sub_choice}. "
                f"{persona_line} "
                "Write only the comment—no greetings, sign‐offs, summaries, or meta-statements. "
                "Do not start your comment with phrases like 'Just', 'Here', 'As a', 'I am', 'This is', "
                "'I'm', greetings, or phrases like 'I've been', 'I spent', 'I just', 'I recently', or similar. "
                "Start directly with a statement, opinion, detail, or description—avoid Reddit formulaic openers. "
                "Try not to use the same opener or sentence structure as earlier comments. Use unique phrasings. "
                "Sound casual and authentic, like a real Redditor. "
                "If the persona is married, do not always mention or focus on your spouse—write as you would on Reddit. "
                "Do not end your comment by asking 'anyone else?' or similar."
            )
            if (idx + 1) % 10 == 0:
                prompt_body += " " + self.diversity_instruction

            full_prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are a helpful assistant that writes realistic, Reddit-style comments based on persona and subreddit context.\n"
                "Your outputs should strictly follow the instructions.\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{prompt_body}\n"
                f"{few_shot_block}"
                "You should imitate the examples I have provided, but you cannot simply modify or rewrite the examples I have given."
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )

            comment = ""
            for _ in range(6):
                out = self.llm(full_prompt, **self.gen_config)
                text = self.clean_comment(out['choices'][0]['text'])
                lower = text.lower()
                bad = (
                    not text or
                    any(lower.startswith(s) for s in self.banned_starts) or
                    any(b in lower for b in self.banned_contains) or
                    any(lower.endswith(e) for e in self.banned_ends) or
                    len(text.split()) < 15
                )
                if not bad:
                    comment = text
                    break

            if not comment:
                comment = "[No valid comment generated]"

            results.append({
                'user_id': row.get('user_id', ''),
                'post_id': row.get('post_id', ''),
                'Generated Text': comment
            })

            # reset KV-cache after every reset interval
            if (idx + 1) % self.reset_interval == 0:
                self.llm.reset()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

        out_df = pd.DataFrame(results)
        outfn = f"/home/yc663354/testing/GenerationScripts/CONTROLScripts/FINALVERSION_5400_CONTROL_META_Inferred_Few_Shot.csv"
        out_df.to_csv(outfn, index=False)
        print(f"Saved {len(out_df)} rows to {outfn}")
        return out_df

if __name__ == "__main__":
    gen = PersonaFewShotGenerator(max_personas=5400)
    df = gen.process_all_personas()
    print(df.head())
