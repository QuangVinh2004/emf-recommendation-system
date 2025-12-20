import pandas as pd
import os

# =====================
# Ensure output folder
# =====================
os.makedirs("data/raw", exist_ok=True)

# =====================
# Convert ratings
# =====================
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)

ratings = ratings[["user_id", "item_id", "rating"]]
ratings.to_csv("data/raw/ratings.csv", index=False)

print("ratings.csv saved:", ratings.shape)

# =====================
# Convert item features
# =====================
genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

items = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None
)

item_features = items[[0] + list(range(5, 24))]
item_features.columns = ["item_id"] + genre_cols

item_features.to_csv("data/raw/item_features.csv", index=False)

print("item_features.csv saved:", item_features.shape)
