from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

corr_matrix = housing.corr(numeric_only=True) # 열들 간의 상관계수를 계산

print(corr_matrix["median_house_value"].sort_values(ascending=False))	#"median_house_value" 열의 상관계수를 추출한 후, 내림차순으로 정렬

housing = strat_train_set.drop("median_house_value", axis=1) # 하우징에 집값을 뺀 것 넣음/ drop은 strat_train_set에 영향을 주지 않음
housing_labels = strat_train_set["median_house_value"].copy() # 집값만 따로 하우징 라벨스에 넣음

# housing.dropna(subset=["total_bedrooms"], inplace=True)    # option 1 해당 구역 제거

# housing.drop("total_bedrooms", axis=1)       # option 2 전체 특성 제거

# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median) # option 3 중간값 대체

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # 누락된 값을 중간값으로 대체하는 객체 생성

housing_num = housing.select_dtypes(include=[np.number]) #중간값은 수치형 특성에서만 계산되므로 수치 특성만 가진 하우스 넘을 생성

imputer.fit(housing_num) # 임퓨터 객체의 fit을 사용해 훈련 데이터에 적용

print(imputer.statistics_)






