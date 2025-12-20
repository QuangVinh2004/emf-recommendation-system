from src.data_loader import load_ratings, load_item_features
from src.model.emf import ExplainableMatrixFactorization


def run_demo():
    # Load dữ liệu
    ratings = load_ratings()
    item_features = load_item_features()

    num_users = ratings["user_id"].nunique()
    num_items = ratings["item_id"].nunique()
    num_features = item_features.shape[1] - 1

    # Train nhanh để demo
    emf = ExplainableMatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        num_features=num_features,
        k=10,
        epochs=20
    )

    emf.train(ratings, item_features)

    print("\n=== DEMO EMF RECOMMENDATION SYSTEM ===")

    while True:
        try:
            user_id = int(input("\nNhập user_id (0 để thoát): "))
            if user_id == 0:
                break

            item_id = int(input("Nhập item_id: "))

            pred = emf.predict(user_id, item_id, item_features)
            print(f"\n👉 Dự đoán rating: {pred:.2f}")

            print("👉 Giải thích:")
            for f, score in emf.explain(user_id, item_id, item_features):
                print(f" - {f}: {score:.3f}")

        except Exception as e:
            print("❌ Lỗi:", e)


if __name__ == "__main__":
    run_demo()
