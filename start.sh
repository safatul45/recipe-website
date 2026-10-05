#!/usr/bin/env bash
set -e

echo "Building recipe data..."
python prepare_data.py
python features.py

echo "Building search model..."
python backend/search.py

echo "Building recommender model..."
python backend/recommender.py

echo "Starting API server..."
uvicorn backend.main:app --host 0.0.0.0 --port $PORT