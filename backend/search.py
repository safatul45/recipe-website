import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Paths ─────────────────────────────────────────────
BASE_DIR      = r"C:\Users\Biostar\Desktop\recipe-website"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_search_model():
    print("Loading featured data...")
    df = pd.read_pickle(os.path.join(PROCESSED_DIR, "recipes_featured.pkl"))
    print("Loaded:", df.shape)

    print("Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = tfidf.fit_transform(df["text_blob"])
    print("TF-IDF shape:", tfidf_matrix.shape)

    with open(os.path.join(MODELS_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    df.to_pickle(os.path.join(MODELS_DIR, "df_search.pkl"))

    print("Saved models to:", MODELS_DIR)
    return tfidf, tfidf_matrix, df


# ─── Cached globals ────────────────────────────────────
_tfidf  = None
_matrix = None
_df     = None


def load_models():
    global _tfidf, _matrix, _df
    if _tfidf is None:
        print("Loading models into memory...")
        with open(os.path.join(MODELS_DIR, "tfidf.pkl"), "rb") as f:
            _tfidf = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb") as f:
            _matrix = pickle.load(f)
        _df = pd.read_pickle(os.path.join(MODELS_DIR, "df_search.pkl"))


def search_recipes(query, top_k=10, filters=None):
    load_models()

    query_vec = _tfidf.transform([query.lower()])
    sims = cosine_similarity(query_vec, _matrix).flatten()

    mask = pd.Series([True] * len(_df))
    if filters:
        if filters.get("cuisine"):
            mask &= _df["cuisine"].isin(filters["cuisine"])
        if filters.get("category"):
            mask &= _df["category"].isin(filters["category"])
        if filters.get("difficulty"):
            mask &= _df["difficulty"].isin(filters["difficulty"])
        if filters.get("max_time"):
            mask &= _df["total_time_minutes"] <= filters["max_time"]
        if filters.get("max_cost"):
            mask &= _df["estimated_cost_usd"] <= filters["max_cost"]
        if filters.get("min_rating"):
            mask &= _df["rating"] >= filters["min_rating"]
        if filters.get("is_vegetarian"):
            mask &= _df["is_vegetarian"] == True
        if filters.get("is_vegan"):
            mask &= _df["is_vegan"] == True

    sims[~mask.values] = -1

    top_indices = sims.argsort()[::-1][:top_k]
    results = _df.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices]

    return results[[
        "recipe_id", "recipe_name", "cuisine", "category",
        "cooking_method", "difficulty", "total_time_minutes",
        "rating", "similarity"
    ]].to_dict(orient="records")


if __name__ == "__main__":
    train_search_model()
    print("\n--- Test search ---")
    results = search_recipes("spicy chicken grill", top_k=5)
    for r in results:
        print(f"  {r['recipe_name']:<25} rating={r['rating']}  sim={r['similarity']:.3f}")