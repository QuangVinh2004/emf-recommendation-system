import numpy as np
import pandas as pd

class ExplainableMatrixFactorization:
    def __init__(self, num_users, num_items, num_features,
                 k=5, lr=0.01, reg=0.01, epochs=50):
        """
        num_users    : số người dùng
        num_items    : số item
        num_features : số feature (Action, Comedy, ...)
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        # Latent factors (MF)
        self.U = np.random.normal(scale=0.1, size=(num_users, k))
        self.V = np.random.normal(scale=0.1, size=(num_items, k))

        # Explainable part
        self.W = np.random.normal(scale=0.1, size=(num_users, num_features))

    def train(self, ratings, item_features):
        """
        ratings: DataFrame(user_id, item_id, rating)
        item_features: DataFrame(item_id, feature1, feature2, ...)
        """
        X = item_features.drop("item_id", axis=1).values

        for epoch in range(self.epochs):
            total_loss = 0

            for _, row in ratings.iterrows():
                u = int(row["user_id"]) - 1
                i = int(row["item_id"]) - 1
                r = row["rating"]

                # Prediction = MF + Explainable
                pred = (
                    np.dot(self.U[u], self.V[i]) +
                    np.dot(self.W[u], X[i])
                )

                error = r - pred

                # Update MF part
                self.U[u] += self.lr * (error * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (error * self.U[u] - self.reg * self.V[i])

                # Update explainable part
                self.W[u] += self.lr * (error * X[i] - self.reg * self.W[u])

                total_loss += error ** 2

            if (epoch + 1) % 10 == 0:
                print(f"[EMF] Epoch {epoch+1}/{self.epochs}, Loss={total_loss:.4f}")

    def predict(self, user_id, item_id, item_features):
        u = user_id - 1
        i = item_id - 1
        X = item_features.drop("item_id", axis=1).values
        return np.dot(self.U[u], self.V[i]) + np.dot(self.W[u], X[i])

    def explain(self, user_id, item_id, item_features, top_k=3):
        """
        Trả về top feature giải thích
        """
        u = user_id - 1
        i = item_id - 1

        feature_names = item_features.columns[1:]
        X = item_features.drop("item_id", axis=1).values

        contributions = self.W[u] * X[i]
        pairs = list(zip(feature_names, contributions))
        pairs.sort(key=lambda x: x[1], reverse=True)

        return pairs[:top_k]
