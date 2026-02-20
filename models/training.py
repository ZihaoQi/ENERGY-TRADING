from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


class ModelTrainer:

    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, y_train, X_test, y_test):

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        print("\nModel Performance")
        print("-" * 50)
        print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")

        return y_pred_test, self.model.predict_proba(X_test)

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model.model, X, y, cv=cv, scoring="accuracy")
        print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores
