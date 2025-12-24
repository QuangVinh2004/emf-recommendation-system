import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from torch.utils.tensorboard import SummaryWriter   # ✅ THÊM

from src.data_loader import load_ratings, load_item_features
from src.model.emf import ExplainableMatrixFactorization


def compute_metrics(model, df, item_features):
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        u = row["user_id"]
        i = row["item_id"]
        r = row["rating"]

        pred = model.predict(u, i, item_features)

        y_true.append(r)
        y_pred.append(pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse


def evaluate_emf(epochs=30):
    # ===== TensorBoard =====
    writer = SummaryWriter(log_dir="logs/emf")  # 👈 KHÔNG cần tạo logs thủ công

    ratings = load_ratings()
    item_features = load_item_features()

    train_df, val_df = train_test_split(
        ratings, test_size=0.2, random_state=42
    )

    model = ExplainableMatrixFactorization(
        num_users=ratings["user_id"].nunique(),
        num_items=ratings["item_id"].nunique(),
        num_features=item_features.shape[1] - 1,
        k=5,
        lr=0.01,
        reg=0.01
    )

    print("Training...\n")

    for epoch in range(epochs):
        start_time = time.time()

        # ===== Train =====
        train_loss = model.train_one_epoch(train_df, item_features)

        # ===== Metrics =====
        train_mse, train_rmse = compute_metrics(model, train_df, item_features)
        val_mse, val_rmse = compute_metrics(model, val_df, item_features)

        elapsed = time.time() - start_time

        # ===== TensorBoard logging =====
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_mse, epoch)

        writer.add_scalar("rmse/train", train_rmse, epoch)
        writer.add_scalar("rmse/val", val_rmse, epoch)

        print(
            f"[*] Epoch: {epoch+1:04d}  "
            f"train_loss= {train_loss:.5f}  "
            f"train_rmse= {train_rmse:.5f}  "
            f"val_loss= {val_mse:.5f}  "
            f"val_rmse= {val_rmse:.5f}  "
            f"time= {elapsed:.5f}"
        )

    writer.close()  # ✅ QUAN TRỌNG


if __name__ == "__main__":
    evaluate_emf()
