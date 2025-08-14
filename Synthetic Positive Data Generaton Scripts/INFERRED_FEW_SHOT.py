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
        embeddings_file: str = '/home/yc663354/testing/data_files/part1bipolar_embedded.csv',
        personas_file: str = '/home/yc663354/testing/GenerationScripts/Final5400MistralPersonas.csv',
        metadata_file: str = '/home/yc663354/testing/GenerationScripts/NEWBipolarMinWord30MaxWord150.csv',
        max_personas: int = None
    ):
        # Normalize file paths
        self.embeddings_file = os.path.abspath(os.path.expanduser(embeddings_file))
        self.personas_file = os.path.abspath(os.path.expanduser(personas_file))
        self.metadata_file = os.path.abspath(os.path.expanduser(metadata_file))
        self.max_personas = max_personas

        # Embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.num_clusters = 200
        self.embeddings_df = None
        self.clusters = None
        self.centroids = None

        # Load metadata containing TID to filter embeddings
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

        # LLM setup: GPU layers for performance
        model_path = os.path.expanduser(
            '/hpcwork/thes2019/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
        )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=35,
            verbose=False
        )

        self.gen_config = {
            "max_tokens": 2048,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": ["<|endoftext|>"]
        }

        # reset interval for KV-cache
        self.reset_interval = 400  # flush cache every 400 generations

    def create_personas_csv(self) -> None:
        print(f"Looking for personas file at: {self.personas_file}")
        if not os.path.exists(self.personas_file):
            raise FileNotFoundError(f"Personas file not found: {self.personas_file}")
        print("Found personas file.")

    def load_embeddings(self) -> np.ndarray:
        import ast
        print(f"Loading embeddings from {self.embeddings_file}...")
        df = pd.read_csv(self.embeddings_file, dtype=str)

        if 'TID' not in df.columns:
            raise ValueError("Embeddings file must contain a 'TID' column as first column")
        df = df[df['TID'].isin(self.valid_tids)].reset_index(drop=True)
        print(f"Filtered embeddings to {len(df)} rows based on metadata TIDs.")

        if 'embedding' not in df.columns:
            raise ValueError(f"Missing 'embedding' column in {self.embeddings_file}")
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        embeddings = np.vstack(df['embedding'].values)

        if 'comment_text' not in df.columns:
            if 'text' in df.columns:
                df['comment_text'] = df['text']
            else:
                df['comment_text'] = [f"Comment {i}" for i in range(len(df))]

        self.embeddings_df = df
        self.cluster_embeddings(embeddings)
        return embeddings

    def cluster_embeddings(self, embeddings: np.ndarray) -> None:
        print(f"Clustering {embeddings.shape[0]} embeddings into {self.num_clusters} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=1000,
            random_state=42,
            n_init=3
        )
        self.clusters = kmeans.fit_predict(embeddings)
        self.centroids = kmeans.cluster_centers_
        self.embeddings_df['cluster'] = self.clusters

    def embed_persona(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def find_cluster_similarities(self, emb: np.ndarray) -> np.ndarray:
        return cosine_similarity(emb.reshape(1, -1), self.centroids)[0]

    def get_cluster_examples(self, cid: int, emb: np.ndarray, num_examples: int = 1):
        sub = self.embeddings_df[self.embeddings_df['cluster'] == cid].copy()
        if sub.empty:
            return []
        arr = np.vstack(sub['embedding'].tolist())
        sims = cosine_similarity(emb.reshape(1, -1), arr)[0]
        sub['sim'] = sims
        sub_sorted = sub.sort_values('sim', ascending=False)
        return sub_sorted.head(num_examples)['comment_text'].tolist()

    def get_cluster_extremes(self, cid: int, emb: np.ndarray):
        sub = self.embeddings_df[self.embeddings_df['cluster'] == cid].copy()
        if sub.empty:
            return []
        arr = np.vstack(sub['embedding'].tolist())
        sims = cosine_similarity(emb.reshape(1, -1), arr)[0]
        sub['sim'] = sims
        sub_sorted = sub.sort_values('sim', ascending=False)
        most_sim = sub_sorted.iloc[0]['comment_text']
        least_sim = sub_sorted.iloc[-1]['comment_text']
        return [most_sim, least_sim]

    def select_diverse_clusters(self, sims: np.ndarray):
        most_sim = np.argmax(sims)
        most_dissim = np.argmin(sims)
        neutral = np.argsort(sims)[len(sims) // 2]
        return most_sim, neutral, most_dissim

    def generate_few_shot_examples(self, emb: np.ndarray) -> list:
        """Generate six examples: most & least similar from top, mid, and bot clusters."""
        sims = self.find_cluster_similarities(emb)
        top, mid, bot = self.select_diverse_clusters(sims)
        examples = []
        for cid in (top, mid, bot):
            extremes = self.get_cluster_extremes(cid, emb)
            examples.extend(extremes)
        return examples

    def generate_comment(self, prompt: str) -> str:
        """
        Use the LLM to generate a comment from the prompt, stripping surrounding quotes if present.
        """
        wrapped = f"[INST] {prompt} [/INST]"
        out = self.llm(wrapped, **self.gen_config)
        if isinstance(out, dict) and 'choices' in out:
            text = out['choices'][0]['text'].strip()
        else:
            text = str(out).strip()
        # Remove surrounding quotes if present
        if (len(text) >= 2) and ((text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'"))):
            text = text[1:-1].strip()
        return text.strip()

    def process_all_personas(self) -> pd.DataFrame:
        self.create_personas_csv()

        with open(self.personas_file, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=[',','    ',';']).delimiter
            except csv.Error:
                delim = ','
        personas_df = pd.read_csv(self.personas_file, sep=delim)
        personas_df.columns = personas_df.columns.str.strip()
        col_map = {c.lower(): c for c in personas_df.columns}
        if 'model_response' in col_map:
            personas_df.rename(columns={col_map['model_response']: 'persona_text'}, inplace=True)
        elif 'persona' in col_map:
            personas_df.rename(columns={col_map['persona']: 'persona_text'}, inplace=True)
        elif 'original_text' in col_map:
            personas_df.rename(columns={col_map['original_text']: 'persona_text'}, inplace=True)
        else:
            raise ValueError("Personas file must contain one of: 'model_response', 'persona', or 'original_text' columns")

        if self.max_personas:
            personas_df = personas_df.head(self.max_personas)

        embeddings = self.load_embeddings()
        results = []
        start = datetime.now()

        data_template = (
        """
Now independently imagine yourself as a normal reddit user and write exactly one reddit comment that would fit in the subreddit {sub_choice}.
Generate only one comment per iteration.
Your comment must be approximately 70-85 words long. Do not use any hashtags in your comment. Do not use greetings like "hello" or exclamations like "wow"
at the start. Exclude preambles. The mention of any explicit reference to bipolar disorder is not allowed.
You must avoid using any language related to mental health such as a condition or general terms like "mental illness",
"diagnosed with", or "suffering from". Be creative with your response. Focus on general behavior, expression, and tone. Do not use a title.
You should imitate the examples I have provided, but you cannot simply modify or rewrite the examples I have given.
        """
        )
        diversity_prompt = (
        """
Provide something more diverse than the previous posts.
Change the structure at the beginning of your response: it shouldn't follow the format of your previous posts.
        """
        )

        for idx, row in tqdm(personas_df.iterrows(), total=len(personas_df), desc="Processing personas"):
            profile = generate_profile(
                occupations=self.occupations_data,
                interests=self.interests_data,
                subreddits=self.subreddits_data,
                countries=self.countries_data
            )
            sub_choice = profile['subreddit']
            examples = self.generate_few_shot_examples(self.embed_persona(row['persona_text']))

            few_shot_block = "Here are example comments for reference:\n\n"
            for i, ex in enumerate(examples, 1):
                few_shot_block += f"{i}. {ex}\n"
            few_shot_block += "\n"

            body = data_template.format(sub_choice=sub_choice)
            ptype = 'normal'
            if (idx + 1) % 5 == 0:
                body += "\n" + diversity_prompt
                ptype = 'diversity'

            persona_line = f"You are a Reddit user with the following persona: {row['persona_text']}. You are diagnosed with Bipolar Disorder."
            full_prompt = f"[INST] {persona_line}\n\n{few_shot_block}{body} [/INST]"

            comment = self.generate_comment(full_prompt)
            results.append({
                'persona_id': idx,
                'user_id': row['user_id'],
                'post_id': row['post_id'],
                'subreddit': sub_choice,
                'persona_text': row['persona_text'],
                'generated_comment': comment,
                'prompt_type': ptype,
                'num_examples_used': len(examples)
            })

            # reset KV-cache after every reset_interval generations
            if (idx + 1) % self.reset_interval == 0:
                self.llm.reset()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

        results_df = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"/home/yc663354/testing/GenerationScripts/AnotherFINAL5400_INFERRED_FEW_SHOT.csv"
        results_df.to_csv(out_file, index=False)

        print(f"Results saved to {out_file}")
        print(f"Processing completed in {datetime.now() - start}")
        return results_df

if __name__ == "__main__":
    gen = PersonaFewShotGenerator(max_personas=5400)
    df = gen.process_all_personas()
    print(df.head())
