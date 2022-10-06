import numpy as np;
data = np.array([12, 34],
                [23, 44],
                [2, 33],
                [3, 55])


data_x = np.fromfunction(lambda _, i: np.power(data[:, 1], i), (1, 4), dtype=float)

print(data_x)