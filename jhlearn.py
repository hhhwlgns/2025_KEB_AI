import numpy as np

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.GDP_list = []
        self.life_list = []

    def fit(self, X, y):
        self.GDP_list = X
        self.life_list = y


    def predict(self, X):
        between = {}

        for i in range (len(self.GDP_list)):
            distance = float(abs(self.GDP_list[i] - X))
            between.update({distance : self.life_list[i]})
        sorted_between = dict(sorted(between.items()))
        sorted_between_list = list(sorted_between.values())
        small_between_y = sorted_between_list[:self.n_neighbors]


        return sum(small_between_y) / self.n_neighbors


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