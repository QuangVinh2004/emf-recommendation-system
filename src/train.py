from src.data_loader import load_ratings, load_item_features
from src.model.emf import ExplainableMatrixFactorization

from sklearn.model_selection import train_test_split
import numpy as np

# =========================
# 1. Load dữ liệu
# =========================
ratings = load_ratings()
item_features = load_item_features()

num_users = ratings["user_id"].nunique()
num_items = ratings["item_id"].nunique()
num_features = item_features.shape[1] - 1  # bỏ cột item_id

# =========================
# 2. Khởi tạo mô hình EMF
# =========================
emf = ExplainableMatrixFactorization(
    num_users=num_users,
    num_items=num_items,
    num_features=num_features,
    k=3,
    epochs=50
)

# =========================
# 3. ĐÁNH GIÁ (MAE & RMSE)
# =========================
train_df, test_df = train_test_split(
    ratings, test_size=0.2, random_state=42
)

emf.train(train_df, item_features)

y_true = []
y_pred = []

for _, row in test_df.iterrows():
    pred = emf.predict(
        row["user_id"],
        row["item_id"],
        item_features
    )
    y_true.append(row["rating"])
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print("\n=== ĐÁNH GIÁ MÔ HÌNH ===")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

# =========================
# 4. DEMO GIẢI THÍCH (CASE STUDY)
# =========================
user_id = 1
item_id = 3

print("\n=== DEMO GIẢI THÍCH ===")
print("Dự đoán rating:", emf.predict(user_id, item_id, item_features))
print("Giải thích:")
for f, score in emf.explain(user_id, item_id, item_features):
    print(f" - {f}: {score:.3f}")
