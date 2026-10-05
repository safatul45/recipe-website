import pandas as pd
import numpy as np
import os

# ─── Paths ─────────────────────────────────────────────
BASE_DIR      = r"C:\Users\Biostar\Desktop\recipe-website"
DATA_DIR      = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Loading CSVs...")
recipes     = pd.read_csv(os.path.join(DATA_DIR, "recipes_master.csv"))
meta        = pd.read_csv(os.path.join(DATA_DIR, "cuisine_metadata.csv"))
ingredients = pd.read_csv(os.path.join(DATA_DIR, "recipe_ingredients.csv"))
steps       = pd.read_csv(os.path.join(DATA_DIR, "recipe_steps.csv"))
nutrition   = pd.read_csv(os.path.join(DATA_DIR, "recipe_nutrition.csv"))

print(f"  recipes:     {recipes.shape}")
print(f"  ingredients: {ingredients.shape}")
print(f"  steps:       {steps.shape}")
print(f"  nutrition:   {nutrition.shape}")

# ─── Clean recipes ─────────────────────────────────────
num_cols = ["prep_time_minutes", "cook_time_minutes", "total_time_minutes",
            "servings", "rating", "review_count", "calories_per_serving",
            "estimated_cost_usd"]
for col in num_cols:
    if col in recipes.columns:
        recipes[col] = recipes[col].fillna(recipes[col].median())

cat_cols = ["cuisine", "category", "cooking_method", "difficulty",
            "spice_level", "meal_type"]
for col in cat_cols:
    if col in recipes.columns:
        recipes[col] = recipes[col].fillna("Unknown")

bool_cols = ["is_vegetarian", "is_vegan", "is_gluten_free", "is_halal",
             "is_traditional", "is_festival_special"]
for col in bool_cols:
    if col in recipes.columns:
        recipes[col] = recipes[col].fillna(False)

recipes["date_added"] = pd.to_datetime(recipes["date_added"], errors="coerce")
recipes["recipe_id"]  = recipes["recipe_id"].astype(str)
ingredients["recipe_id"] = ingredients["recipe_id"].astype(str)
steps["recipe_id"]       = steps["recipe_id"].astype(str)
nutrition["recipe_id"]   = nutrition["recipe_id"].astype(str)

# ─── Aggregate ingredients per recipe ──────────────────
print("Aggregating ingredients...")
def aggregate_ingredients(group):
    names = group["ingredient_name"].dropna().tolist()
    seen, unique_names = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique_names.append(n)
    return pd.Series({
        "ingredient_list":       unique_names,
        "ingredient_count":      len(unique_names),
        "ingredient_categories": list(set(group["category"].dropna().tolist()))
    })

ing_agg = ingredients.groupby("recipe_id").apply(aggregate_ingredients).reset_index()

# ─── Aggregate steps per recipe ────────────────────────
print("Aggregating steps...")
def aggregate_steps(group):
    group = group.sort_values("step_number")
    return pd.Series({
        "steps":      group["step_description"].dropna().tolist(),
        "step_count": len(group),
        "steps_time": int(group["estimated_time_minutes"].fillna(0).sum())
    })

steps_agg = steps.groupby("recipe_id").apply(aggregate_steps).reset_index()

# ─── Merge everything ──────────────────────────────────
print("Merging...")
df = recipes.merge(ing_agg, on="recipe_id", how="left")
df = df.merge(steps_agg, on="recipe_id", how="left")
df = df.merge(nutrition, on="recipe_id", how="left", suffixes=("", "_nutr"))

df["ingredient_list"]       = df["ingredient_list"].apply(lambda x: x if isinstance(x, list) else [])
df["ingredient_categories"] = df["ingredient_categories"].apply(lambda x: x if isinstance(x, list) else [])
df["steps"]                 = df["steps"].apply(lambda x: x if isinstance(x, list) else [])
df["ingredient_count"]      = df["ingredient_count"].fillna(0)
df["step_count"]            = df["step_count"].fillna(0)

print("Final shape:", df.shape)
out = os.path.join(PROCESSED_DIR, "recipes_clean.pkl")
df.to_pickle(out)
print("Saved:", out)
print("\nColumns:")
for c in df.columns:
    print("  -", c)