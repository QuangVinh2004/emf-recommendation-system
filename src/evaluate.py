import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_ratings, load_item_features
from src.model.emf import ExplainableMatrixFactorization


def evaluate_emf():
    # =========================
    # 1. Load dữ liệu
    # =========================
    ratings = load_ratings()
    item_features = load_item_features()

    num_users = ratings["user_id"].nunique()
    num_items = ratings["item_id"].nunique()
    num_features = item_features.shape[1] - 1

    # =========================
    # 2. Train / Test split
    # =========================
    train_df, test_df = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

    print("Train size:", train_df.shape)
    print("Test size :", test_df.shape)

    # =========================
    # 3. Khởi tạo EMF
    # =========================
    emf = ExplainableMatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        num_features=num_features,
        k=10,
        epochs=30
    )

    # =========================
    # 4. Train
    # =========================
    emf.train(train_df, item_features)

    # =========================
    # 5. Dự đoán trên test set
    # =========================
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]
        true_rating = row["rating"]

        pred_rating = emf.predict(user_id, item_id, item_features)

        y_true.append(true_rating)
        y_pred.append(pred_rating)

    # =========================
    # 6. Tính MAE & RMSE
    # =========================
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    print("\n====== ĐÁNH GIÁ MÔ HÌNH EMF ======")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")


if __name__ == "__main__":
    evaluate_emf()
