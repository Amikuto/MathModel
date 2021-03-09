import numpy as np

"""
Formula 1
"""
P = np.array([
    [0.4, 0.3, 0.3],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7]
])

k = 2
out1 = np.linalg.matrix_power(P, k)
# print(out1)


"""
Formula 2
"""
P = np.array([
    [0.4, 0.3, 0.3],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7]
])
A0 = np.array([0.5, 0, 0.5])

k = 2
out2 = np.outer(A0, np.linalg.matrix_power(P, k))
# print(out2)


"""
Formula 3
"""

P = np.array([
    [0.4, 0.3, 0.3],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7]
])

i = 1
j = 3
k = 3
m = j


for o in range():
    pass

for t in P:
    print(t)

for i in range():
    pass
for t in range(P[0]):
    pass