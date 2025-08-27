import numpy as np
from sklearn.linear_model import LinearRegression


def train_model():
    np.random.seed(42)
    X = np.array([
        [5, 1],
        [10, 2],
        [20, 3],
        [15, 1],
        [25, 2],
        [30, 3],
        [12, 1],
        [18, 2],
        [22, 3],
        [8, 1]
    ])
    true_eta = np.dot(X, np.array([2, 10])) + 5
    noise = np.random.normal(0, 5, size=true_eta.shape)

    y = true_eta + noise


    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    return model

def predict(model, features):
    new_delivery = np.array([[features[0], features[1]]])
    return model.predict(new_delivery)