import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None


    def fit(self, X, y):
        """
        훈련 함수
        :param X: 변수
        :param y: 값
        :return:
        """
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        denominator = np.sum(pow(X-X_mean, 2))
        numerator = np.sum((X-X_mean)*(y-y_mean))

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * X_mean)


    def predict(self, X) -> np.ndarray:
        """
        예측 함수
        :param X: 새로운 변수
        :return: 예측 값
        """
        return self.slope * np.array(X) + self.intercept