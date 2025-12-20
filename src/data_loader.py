import pandas as pd

def load_ratings(path="data/raw/ratings.csv"):
    return pd.read_csv(path)

def load_item_features(path="data/raw/item_features.csv"):
    return pd.read_csv(path)
