import numpy as np
m = 3
n = 16
nu = 16
lam = 31


"""
Задание a
"""
L = np.zeros((m+n+1, m+n+1))

count = 0

for i in range(m + n + 1):
    for j in range(m + n + 1):
        if i == (j - 1):
            L[i][j] = lam
        if j == (i - 1) and i < m + 1:
            L[i][j] = nu * count
        if j == (i - 1) and i > m:
            L[i][j] = max(L[m])
    count += 1

# print(L)

L_transpose = np.transpose(L)

D = np.zeros((m+n+1, m+n+1))
for i in range(m + n + 1):
    for j in range(m + n + 1):
        if i == j:
            D[i][j] = sum(L[i])

M = L_transpose - D
# print(M)

B = np.zeros((m+n+1))
B[m + n] = 1

# print(B)

M_ = M
M_[m + n] = 1

# print(M_)

X = np.linalg.matrix_power(M_, -1) @ B

print("Задание a: ", X)

"""
Задание b
"""

print("Задание b: ", X[-1])


"""
Задание c
"""

otn_matrix = 1 - X[-1]
print("Задание c: ", "\nОтносительная: ", otn_matrix, "\nАбсолютная: ", otn_matrix * lam, "\n")


"""
Задание d
"""

out = 0
for i in range(1, n + 1):
    out += i * X[m + i]
print("Задание d: ", out)


"""
Задание e
"""

c = 0
for i in range(n):
    c += ((i + 1) / (m * nu) * X[m + i])
print("Задание e: ", c)



"""
Задание f
"""


c = 0
for i in range(1, m + n + 1):
    if i <= m:
        c += i * X[i]
    else:
        c += m * X[i]
print("Задание f: ", c)



"""
Задание g
"""

print("Задание g: ", np.nansum(X[:m]))


"""
Задание h
"""


print("Задание h: ", 1 / np.nansum(L, -1)[0])
