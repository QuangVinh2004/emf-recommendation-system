from src.data_loader import load_ratings, load_item_features
from src.model.emf import ExplainableMatrixFactorization

def main():
    # Load dữ liệu
    ratings = load_ratings()
    item_features = load_item_features()

    num_users = ratings["user_id"].nunique()
    num_items = ratings["item_id"].nunique()
    num_features = item_features.shape[1] - 1

    # Khởi tạo EMF
    emf = ExplainableMatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        num_features=num_features,
        k=10,
        epochs=50
    )

    # Train
    emf.train(ratings, item_features)

    print("✔ Training completed.")

if __name__ == "__main__":
    main()
