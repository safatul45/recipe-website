from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import os

from backend.search import search_recipes
from backend.recommender import recommend_similar, recommend_by_query

BASE_DIR      = r"C:\Users\Biostar\Desktop\recipe-website"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

app = FastAPI(title="Recipe API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # keep for now — tighten later
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading featured data for API...")
df = pd.read_pickle(os.path.join(PROCESSED_DIR, "recipes_featured.pkl"))
print("Loaded:", df.shape)


class RecipeCard(BaseModel):
    recipe_id: str
    recipe_name: str
    cuisine: str
    category: str
    cooking_method: str
    difficulty: str
    total_time_minutes: int
    rating: float
    calories_per_serving: int


@app.get("/")
def root():
    return {"message": "Recipe API running", "total_recipes": len(df)}


@app.get("/health")
def health():
    return {"status": "ok", "total_recipes": len(df)}


@app.get("/recipes", response_model=List[RecipeCard])
def list_recipes(
    cuisine: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    max_time: Optional[int] = None,
    max_cost: Optional[float] = None,
    min_rating: Optional[float] = None,
    vegetarian: Optional[bool] = None,
    vegan: Optional[bool] = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
):
    result = df
    if cuisine:    result = result[result["cuisine"] == cuisine]
    if category:   result = result[result["category"] == category]
    if difficulty: result = result[result["difficulty"] == difficulty]
    if max_time:   result = result[result["total_time_minutes"] <= max_time]
    if max_cost:   result = result[result["estimated_cost_usd"] <= max_cost]
    if min_rating: result = result[result["rating"] >= min_rating]
    if vegetarian is not None: result = result[result["is_vegetarian"] == vegetarian]
    if vegan is not None:      result = result[result["is_vegan"] == vegan]

    result = result.iloc[offset: offset + limit]
    return result[[
        "recipe_id", "recipe_name", "cuisine", "category",
        "cooking_method", "difficulty", "total_time_minutes",
        "rating", "calories_per_serving"
    ]].to_dict(orient="records")


@app.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: str):
    matches = df[df["recipe_id"] == recipe_id]
    if matches.empty:
        raise HTTPException(404, "Recipe not found")
    row = matches.iloc[0]
    return {
        "recipe_id": row["recipe_id"],
        "recipe_name": row["recipe_name"],
        "cuisine": row["cuisine"],
        "category": row["category"],
        "cooking_method": row["cooking_method"],
        "difficulty": row["difficulty"],
        "prep_time_minutes": int(row["prep_time_minutes"]),
        "cook_time_minutes": int(row["cook_time_minutes"]),
        "total_time_minutes": int(row["total_time_minutes"]),
        "servings": int(row["servings"]),
        "spice_level": row["spice_level"],
        "meal_type": row["meal_type"],
        "is_vegetarian": bool(row["is_vegetarian"]),
        "is_vegan": bool(row["is_vegan"]),
        "is_gluten_free": bool(row["is_gluten_free"]),
        "is_halal": bool(row["is_halal"]),
        "rating": float(row["rating"]),
        "review_count": int(row["review_count"]),
        "calories_per_serving": int(row["calories_per_serving"]),
        "estimated_cost_usd": float(row["estimated_cost_usd"]),
        "ingredients": row["ingredient_list"],
        "steps": row["steps"],
        "nutrition": {
            "calories": float(row.get("calories", 0) or 0),
            "protein_g": float(row.get("protein_g", 0) or 0),
            "carbohydrates_g": float(row.get("carbohydrates_g", 0) or 0),
            "fat_g": float(row.get("fat_g", 0) or 0),
            "fiber_g": float(row.get("fiber_g", 0) or 0),
            "sugar_g": float(row.get("sugar_g", 0) or 0),
            "sodium_mg": float(row.get("sodium_mg", 0) or 0),
            "cholesterol_mg": float(row.get("cholesterol_mg", 0) or 0),
        },
    }


@app.get("/search")
def search(
    q: str,
    top_k: int = 10,
    cuisine: Optional[str] = None,
    max_time: Optional[int] = None,
    min_rating: Optional[float] = None,
    vegetarian: Optional[bool] = None,
):
    filters = {}
    if cuisine: filters["cuisine"] = [cuisine]
    if max_time: filters["max_time"] = max_time
    if min_rating: filters["min_rating"] = min_rating
    if vegetarian is not None: filters["is_vegetarian"] = vegetarian

    return search_recipes(q, top_k=top_k, filters=filters or None)


@app.get("/recipes/{recipe_id}/similar")
def similar(recipe_id: str, top_k: int = 5):
    return recommend_similar(recipe_id, top_k=top_k)


@app.get("/recommend")
def recommend(q: str, top_k: int = 5):
    return recommend_by_query(q, top_k=top_k)