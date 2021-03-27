import numpy as np
m = 3
n = 16
nu = 16
# u = 2
lam = 31
# L = np.zeros((m, n))

L = np.zeros((m+n+1, m+n+1))

# for i in range(m+1):
#     for j in range(n+1):
#         if i == j or abs(i-j) > 1:
#             L[i][j] = 0
#         elif i > j:
#             L[i][j] = min([m, i]) * u
#         else:
#             L[i][j] = 1

# print(L)

count = 0
# for i in range(m+1):
#     for j in range(n+1):
#         if i == (j - 1):
#             L[i][j] = 1
#         if j == (i-1):
#             L[i][j] = nu * count
#     count += 1

for i in range(m + n + 1):
    for j in range(m + n + 1):
        if i == (j - 1):
            L[i][j] = lam
        if j == (i - 1) and i < m + 1:
            L[i][j] = nu * count
        if j == (i - 1) and i > m:
            L[i][j] = max(L[m]) * nu
    count += 1

print(L)

L_transpose = np.transpose(L)

D = np.zeros((m+n+1, m+n+1))
for i in range(m + n + 1):
    for j in range(m + n + 1):
        if i == j:
            D[i][j] = sum(L[i])

M = L_transpose - D

B = np.zeros((m+n+1))
B[m + n] = 1

print(B)

M_ = M
M_[m + n] = 1

print(M_)

X = np.linalg.matrix_power(M_, -1) @ B

print(sum(X))
