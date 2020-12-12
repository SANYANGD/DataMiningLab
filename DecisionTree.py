import numpy as np
import math


# 计算香农熵 pklog2pk
def countH(m, n):
    if m == 0 or n == 0: return 0
    return -m / (m + n) * math.log(m / (m + n), 2) - n / (m + n) * math.log(n / (m + n), 2)


dataset = [[21, 37, 0],
           [27, 41, 0],
           [43, 61, 1],
           [38, 55, 0],
           [44, 30, 0],
           [51, 56, 1],
           [53, 70, 1],
           [56, 74, 1],
           [59, 25, 0],
           [61, 68, 1],
           [63, 51, 1]]
dataset = np.array(dataset)
ent = 0
for i in range(0, 11):
    if dataset[i, 2] == 0:
        ent = ent + 1
ent = countH(ent, 11 - ent)
salarySorted = dataset[dataset[:, 1].argsort()]
ageSorted = dataset[dataset[:, 0].argsort()]


def Gain(Sort):
    for i in range(1, 11):
        a = 0
        b = 0
        for j in range(0, i):
            if Sort[j, 2] == 0:
                a = a + 1
        a = countH(a, i - a)
        for j in range(i, 11):
            if Sort[j, 2] == 0:
                b = b + 1
        b = countH(b, 11 - i - b)
        c = ent - (a * i / 11 + b * (11 - i) / 11)
        print(c)


if __name__ == '__main__':
    Gain(ageSorted)
    print('\n')
    Gain(salarySorted)
    print(ageSorted)
    print(salarySorted)

