import numpy as np


class MyLogisticRegression:

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def p_initialization(self):
        self.W = np.zeros(self.X.shape[1])
        self.b = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -35, 35)
        return 1 / (1 + np.exp(-z))

    def foward(self, X):
        linear_combination = np.matmul(X, self.W) + self.b
        activation = self.sigmoid(linear_combination)
        return activation

    def gradient(self, X, predictions):
        m = X.shape[0]
        dW = (1 / m) * np.matmul(X.T, (predictions - self.y))
        db = (1 / m) * np.sum(predictions - self.y)
        return dW, db

    def cost_function(self, prediction):
        m = self.X.shape[0]
        eps = 1e-12
        prediction = np.clip(prediction, eps, 1 - eps)
        return (1 / m) * np.sum(
            -(self.y * np.log(prediction) + (1 - self.y) * np.log(1 - prediction))
        )

    def gradient_descent(self, iterations):
        self.iterations = iterations
        self.p_initialization()
        self.y = np.asarray(self.y, dtype=float).reshape(
            -1,
        )
        self.X = np.asarray(self.X, dtype=float)

        for _ in range(self.iterations):
            predictions = self.foward(self.X)

            cost = self.cost_function(predictions)

            dW, db = self.gradient(self.X, predictions)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def hessian(self, X, predictions):
        X = np.asarray(X, dtype=float)
        p = np.asarray(predictions, dtype=float)

        m, n = X.shape
        z = p * (1 - p)
        Xw = X * z[:, None]

        Hw = (1 / m) * np.matmul(X.T, Xw)
        Hb = np.mean(z) + 1e-12

        return Hw, Hb

    def newton_method(self, iterations):
        self.p_initialization()
        self.iterations = iterations

        X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        for _ in range(self.iterations):
            predictions = self.foward(X)
            dW, db = self.gradient(X, predictions)
            Hw, Hb = self.hessian(X, predictions)

            # delta_w = np.linalg.solve(Hw, dW)#pseudo-inversa ca sa mearga si daca nu e inversabila hessiana(in cazul in care mofifica seed din train_test_split)
            delta_w = np.linalg.pinv(Hw) @ dW
            delta_b = db / Hb

            self.W -= delta_w
            self.b -= delta_b

            if np.linalg.norm(delta_w) < 1e-6 and abs(delta_b) < 1e-6:
                break

    def predict(self, X):
        predictions = self.foward(np.asarray(X, dtype=float))
        return (predictions >= 0.5).astype(int)
