import numpy as np


class ExplainableMatrixFactorization:
    def __init__(self, num_users, num_items, num_features,
                 k=5, lr=0.01, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.k = k
        self.lr = lr
        self.reg = reg

        # MF latent factors
        self.U = np.random.normal(scale=0.1, size=(num_users, k))
        self.V = np.random.normal(scale=0.1, size=(num_items, k))

        # Explainable user-feature weights
        self.W = np.random.normal(scale=0.1, size=(num_users, num_features))

    def predict(self, user_id, item_id, item_features):
        u = user_id - 1
        i = item_id - 1
        X = item_features.drop("item_id", axis=1).values
        return np.dot(self.U[u], self.V[i]) + np.dot(self.W[u], X[i])

    def train_one_epoch(self, ratings, item_features):
        X = item_features.drop("item_id", axis=1).values
        total_loss = 0.0

        for _, row in ratings.iterrows():
            u = int(row["user_id"]) - 1
            i = int(row["item_id"]) - 1
            r = row["rating"]

            pred = np.dot(self.U[u], self.V[i]) + np.dot(self.W[u], X[i])
            err = r - pred

            # Update MF
            self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
            self.V[i] += self.lr * (err * self.U[u] - self.reg * self.V[i])

            # Update explainable part
            self.W[u] += self.lr * (err * X[i] - self.reg * self.W[u])

            total_loss += err ** 2

        return total_loss / len(ratings)

    def explain(self, user_id, item_id, item_features, top_k=3):
        u = user_id - 1
        i = item_id - 1

        feature_names = item_features.columns[1:]
        X = item_features.drop("item_id", axis=1).values

        contributions = self.W[u] * X[i]
        pairs = list(zip(feature_names, contributions))
        pairs.sort(key=lambda x: x[1], reverse=True)

        return pairs[:top_k]
