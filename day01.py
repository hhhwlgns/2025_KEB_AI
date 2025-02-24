import numpy as np
import pandas as pd

df = pd.DataFrame(
    {"a":[4,5,6],
     "b":[7,8,9],
     "c":[10,11,12]}, index = [1, 2, 3]
)

print(df)

nf = pd.DataFrame(
    [[4, 7, 10],
     [5, 8, 11],
     [6, 9, 12]],
     index = [1, 2, 3], columns = ['a', 'b', 'c']
)

print(nf)