import pandas as pd
import os

BASE_DIR      = r"C:\Users\Biostar\Desktop\recipe-website"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

print("Loading cleaned data...")
df = pd.read_pickle(os.path.join(PROCESSED_DIR, "recipes_clean.pkl"))
print("Loaded:", df.shape)

# ─── Build text blob for search ────────────────────────
def build_text_blob(row):
    parts = [
        str(row.get("recipe_name", "")),
        str(row.get("cuisine", "")),
        str(row.get("category", "")),
        str(row.get("cooking_method", "")),
        str(row.get("difficulty", "")),
        str(row.get("spice_level", "")),
        str(row.get("meal_type", "")),
    ]
    if isinstance(row.get("ingredient_list"), list):
        parts.extend(row["ingredient_list"])
    if isinstance(row.get("ingredient_categories"), list):
        parts.extend(row["ingredient_categories"])
    return " ".join([p.lower() for p in parts if p and p.lower() != "nan"])

print("Building text blob...")
df["text_blob"] = df.apply(build_text_blob, axis=1)

# ─── Numeric features ──────────────────────────────────
print("Computing numeric features...")
df["cost_per_serving"]    = df["estimated_cost_usd"] / df["servings"].replace(0, 1)
df["calories_per_dollar"] = df["calories_per_serving"] / df["estimated_cost_usd"].replace(0, 0.01)
df["prep_ratio"]          = df["prep_time_minutes"] / df["total_time_minutes"].replace(0, 1)

def time_bucket(mins):
    if mins <= 30:  return "very_quick"
    if mins <= 60:  return "quick"
    if mins <= 120: return "medium"
    return "long"

df["time_bucket"]   = df["total_time_minutes"].apply(time_bucket)
df["is_quick"]      = df["total_time_minutes"] < 60
df["is_cheap"]      = df["cost_per_serving"] < 3.0
df["is_high_rated"] = df["rating"] >= 4.5
df["is_popular"]    = df["review_count"] >= 2000

# ─── Save ──────────────────────────────────────────────
out = os.path.join(PROCESSED_DIR, "recipes_featured.pkl")
df.to_pickle(out)
print("Saved:", out)
print("\nSample text_blob (first 2 recipes):")
for i in range(2):
    print(f"  [{df.iloc[i]['recipe_name']}] {df.iloc[i]['text_blob'][:120]}...")
print("\nNew columns added:")
print("  text_blob, cost_per_serving, calories_per_dollar, prep_ratio,")
print("  time_bucket, is_quick, is_cheap, is_high_rated, is_popular")