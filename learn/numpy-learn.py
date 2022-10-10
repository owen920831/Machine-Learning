import math
data_x = 0
for n in range(21, 121):
    data_x += math.comb(120, n)*math.pow(0.1, n)*math.pow(0.9, 120-n)

print(data_x)