import numpy as np

class LinearRegression():
    def __init__(self, lr=0.01, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.slope = None
        self.y_intercept = None

    def fit(self, X, y) -> None:
        # Initalize the Slope and y-intercept as 0
        n_samples, n_features = X.shape
        self.slope = np.zeros(n_features)
        self.y_intercept = 0

        for _ in range(self.n_iters):
            # Predict the line using y=ax+b
            y_pred = np.dot(X, self.slope) + self.y_intercept

            # Calculate the error and use Gradient descent
            da = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.slope = self.slope - self.lr * da
            self.y_intercept = self.y_intercept - self.lr * db

    # Testing
    def predict(self, X):
        if self.slope is None:
            raise ValueError("Model must be fitted before predicting")
        y_pred = np.dot(X, self.slope) + self.y_intercept
        return y_pred
