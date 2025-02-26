from cProfile import label

import pandas as pd

data = [1,7,5,2,8,3,6,4]

bins = [0,3,6,9]

labels = ["low", "middle", "high"]

cat = pd.cut(data, bins, True, labels)

print(cat)