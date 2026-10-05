from backend.dll_fix import fix_torch_dll
fix_torch_dll()

import pandas as pd
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
BASE_DIR      = r"C:\Users\Biostar\Desktop\recipe-website"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_embeddings():
    print("Loading featured data...")
    df = pd.read_pickle(os.path.join(PROCESSED_DIR, "recipes_featured.pkl"))
    print("Loaded:", df.shape)

    print("Loading SentenceTransformer (first run downloads ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding recipes (this may take a few minutes)...")
    embeddings = model.encode(
        df["text_blob"].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    embeddings = np.array(embeddings).astype("float32")
    print("Embeddings shape:", embeddings.shape)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(MODELS_DIR, "recipe_index.faiss"))
    np.save(os.path.join(MODELS_DIR, "recipe_embeddings.npy"), embeddings)
    df.to_pickle(os.path.join(MODELS_DIR, "df_recommender.pkl"))

    print(f"Built FAISS index: {index.ntotal} recipes, dim={dim}")


# ─── Cached globals ────────────────────────────────────
_index      = None
_embeddings = None
_df         = None
_model      = None


def load_recommender():
    global _index, _embeddings, _df, _model
    if _index is None:
        print("Loading recommender into memory...")
        _index      = faiss.read_index(os.path.join(MODELS_DIR, "recipe_index.faiss"))
        _embeddings = np.load(os.path.join(MODELS_DIR, "recipe_embeddings.npy"))
        _df         = pd.read_pickle(os.path.join(MODELS_DIR, "df_recommender.pkl"))
        _model      = SentenceTransformer("all-MiniLM-L6-v2")


def recommend_similar(recipe_id, top_k=5):
    load_recommender()

    matches = _df[_df["recipe_id"] == recipe_id]
    if matches.empty:
        return []
    idx = matches.index[0]

    vec = _embeddings[idx:idx + 1]
    distances, indices = _index.search(vec, top_k + 1)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i == idx:
            continue
        row = _df.iloc[i]
        results.append({
            "recipe_id":   row["recipe_id"],
            "recipe_name": row["recipe_name"],
            "cuisine":     row["cuisine"],
            "rating":      float(row["rating"]),
            "similarity":  float(1 / (1 + dist))
        })
        if len(results) >= top_k:
            break
    return results


def recommend_by_query(query, top_k=5):
    load_recommender()

    q_emb = _model.encode([query]).astype("float32")
    distances, indices = _index.search(q_emb, top_k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        row = _df.iloc[i]
        results.append({
            "recipe_id":   row["recipe_id"],
            "recipe_name": row["recipe_name"],
            "rating":      float(row["rating"]),
            "similarity":  float(1 / (1 + dist))
        })
    return results


if __name__ == "__main__":
    build_embeddings()

    print("\n--- Test: similar to RCP00001 ---")
    for r in recommend_similar("RCP00001"):
        print(f"  {r['recipe_name']:<25} rating={r['rating']}  sim={r['similarity']:.3f}")

    print("\n--- Test: query 'paneer spinach curry' ---")
    for r in recommend_by_query("paneer spinach curry"):
        print(f"  {r['recipe_name']:<25} rating={r['rating']}  sim={r['similarity']:.3f}")