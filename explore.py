import pandas as pd
import os
# ─── Paths (Windows) ───────────────────────────────────
BASE_DIR = r"C:\Users\Biostar\Desktop\recipe-website"
DATA_DIR = os.path.join(BASE_DIR, "data")
recipes_path     = os.path.join(DATA_DIR, "recipes_master.csv")
meta_path        = os.path.join(DATA_DIR, "cuisine_metadata.csv")
ingredients_path = os.path.join(DATA_DIR, "recipe_ingredients.csv")

# ─── Load ──────────────────────────────────────────────
recipes     = pd.read_csv(recipes_path)
meta        = pd.read_csv(meta_path)
ingredients = pd.read_csv(ingredients_path)

print("Recipes shape:    ", recipes.shape)
print("Meta shape:       ", meta.shape)
print("Ingredients shape:", ingredients.shape)

print("\nRecipes columns:", recipes.columns.tolist())
print("\nFirst 3 rows of recipes:")
print(recipes.head(3))