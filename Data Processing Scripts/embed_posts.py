import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Config
INPUT_CSV = "files/GoogleDriveData/balanced_bipolar_part1 (1).csv"
OUTPUT_CSV = "files/embedded_posts/part1bipolar_embedded.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, good quality

# Load data
df = pd.read_csv(INPUT_CSV)

# Initialize model
model = SentenceTransformer(MODEL_NAME)

# Compute embeddings
embeddings = []
for text in tqdm(df['text'].astype(str).tolist(), desc="Embedding posts"):
    emb = model.encode(text)
    embeddings.append(emb)

embeddings = np.stack(embeddings)
df['embedding'] = embeddings.tolist()

# Save embeddings as string (or npy for efficiency)
df.to_csv(OUTPUT_CSV, index=False)
np.save("embedded_posts.npy", embeddings)  # Optional: more efficient for future loads

print("Done! Embeddings saved.")
