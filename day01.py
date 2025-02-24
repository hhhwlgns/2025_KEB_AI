#from statistics import LinearRegression
from sklearn.linear_model import LinearRegression # 선형 회귀 모델
from sklearn.neighbors import KNeighborsRegressor # 최근접 이웃 회귀 모델
import pandas as pd
import matplotlib.pyplot as plt

# 데이터를 다운로드하고 준비
ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv") # 이 주소에 있는 값을 읽어옴.
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values

# 데이터를 그래프로 나타냄
ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction") # scatter 형태로, grid는 수직선 표시 여부
plt.axis([23500, 62500, 4, 9]) # x는 23500에서 62500까지 y는 4에서 9까지
plt.show()

# 선형 회귀 모델을 선택 1
model1 = LinearRegression()

# 훈련하기 1
model1.fit(X,y)

# 키프로스에 대해 예측을 만들기 1
X_new1 = [[37655.2]] # 2020년 키프로스 1인당 GDP
print(model1.predict(X_new1)) # y값 예측

# 최근접 이웃 회귀 모델을 선택 2
model2 = KNeighborsRegressor(n_neighbors=3) # 가장 가까운 3개 선택 후 평균을 구함

# 훈련하기 2
model2.fit(X, y)

# 키프로스에 대해 예측 만들기 2
X_new2 = [[37655.2]] # 2020년 키프로스 1인당 GDP
print(model2.predict(X_new2)) # y값 예측


