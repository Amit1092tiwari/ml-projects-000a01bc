"""Unit tests for recommender system algorithms."""

import sys
from pathlib import Path
import pandas as pd

# Add recommender directory to sys.path
recommender_dir = Path(__file__).resolve().parent.parent / "Million Songs Dataset - Recommendation Engine"
sys.path.insert(0, str(recommender_dir))

from Recommenders import popularity_recommender_py, item_similarity_recommender_py


def test_popularity_recommender():
    train_data = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u1", "u2", "u1"],
        "song": ["s1", "s1", "s1", "s2", "s2", "s3"],
    })
    rec = popularity_recommender_py()
    rec.create(train_data, "user_id", "song")

    recommendations = rec.recommend("u4")

    assert len(recommendations) == 3
    assert "user_id" in recommendations.columns
    assert recommendations.iloc[0]["song"] == "s1"  # Most popular song (3 plays)
    assert recommendations.iloc[0]["score"] == 3


def test_item_similarity_recommender():
    train_data = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "song": ["s1", "s2", "s1", "s2", "s3"],
    })
    rec = item_similarity_recommender_py()
    rec.create(train_data, "user_id", "song")

    recommendations = rec.recommend("u3")
    assert isinstance(recommendations, pd.DataFrame)
