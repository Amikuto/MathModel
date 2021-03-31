import numpy as np


P = np.array([
    #1     2     3     4     5     6     7     8     9     10    11
    [0.17, 0,    0.83, 0,    0,    0,    0,    0,    0,    0,    0],  # 1
    [0.39, 0.05, 0.24, 0,    0.11, 0.21, 0,    0,    0,    0,    0],  # 2
    [0.22, 0,    0.03, 0.21, 0,    0,    0.18, 0.13, 0.23, 0,    0],  # 3
    [0,    0.09, 0.19, 0.07, 0,    0,    0.49, 0.16, 0,    0,    0],  # 4
    [0,    0,    0,    0.31, 0.04, 0,    0.36, 0.29, 0,    0,    0],  # 5
    [0,    0.05, 0,    0,    0.27, 0.37, 0.27, 0,    0.04, 0,    0],  # 6
    [0,    0,    0.01, 0,    0.42, 0.12, 0.06, 0.27, 0,    0.12, 0],  # 7
    [0,    0,    0,    0,    0,    0,    0,    0.96, 0.04, 0,    0],  # 8
    [0,    0,    0.63, 0,    0,    0.31, 0,    0,    0.06, 0,    0],  # 9
    [0,    0,    0,    0,    0,    0,    0.36, 0,    0.13, 0.19, 0.32],  # 10
    [0,    0,    0,    0,    0,    0,    0.45, 0,    0.30, 0,    0.25],  # 11
])


def skip_function(bf_mrx, matrix, matrix_length):
    new_matrix = np.zeros((matrix_length, matrix_length))
    for i in range(matrix_length):
        for j in range(matrix_length):
            z = 0
            for o in range(matrix_length):
                if o != j:
                    z += bf_mrx[o, j] * matrix[i, o]
            new_matrix[i, j] = z
    return new_matrix


"""
Task 1
"""


def task1(matrix, k, i, j):
    res = np.linalg.matrix_power(matrix, k)
    return res[i - 1][j - 1]


k, i, j = 10, 1, 7
print("Задание 1: ", task1(matrix=P, k=k, i=i, j=j))


"""
Task 2
"""

A = np.array([0.04, 0.04, 0.09, 0.03, 0.11, 0, 0.19, 0.17, 0.12, 0.16, 0.05])


def task2(matrix, k, a_start):
    return a_start @ np.linalg.matrix_power(matrix, k)


k = 10
print("Задание 2: ", task2(matrix=P, k=k, a_start=A))


"""
Task 3
"""


def task3(matrix, k, i, j):
    matrix_out = matrix.copy()
    for c in range(k-1):
        matrix_out = skip_function(bf_mrx=matrix_out, matrix=matrix, matrix_length=len(matrix))
    return matrix_out[i - 1][j - 1]


k, i, j = 8, 8, 7
print("Задание 3: ", task3(matrix=P, k=k, i=i, j=j))


"""
Task 4
"""


def task4(matrix, k, i, j):
    buff_matrix = matrix_out = matrix.copy()
    for c in range(k-1):
        buff_matrix = skip_function(bf_mrx=buff_matrix, matrix=matrix, matrix_length=len(matrix))
        matrix_out += buff_matrix
    return matrix_out[i-1][j-1]


i, j, k = 6, 8, 5
print("Задание 4: ", task4(matrix=P, k=k, i=i, j=j))


"""
Task 5
"""


def task5(matrix, i, j):
    bf_mrx = matrix_out = matrix.copy()
    for t in range(1, 2048):
        bf_mrx = skip_function(bf_mrx=bf_mrx, matrix=matrix, matrix_length=len(matrix))
        matrix_out += t * bf_mrx
    return matrix_out[i-1][j-1]


i, j = 2, 8
print("Задание 5: ", task5(matrix=P, i=i, j=j))


"""
Task 6
"""


def task6(matrix, k, i):
    # x1 = np.linalg.matrix_power(before_matrix, 1)
    # x2 = np.linalg.matrix_power(before_matrix, 2)

    # print(np.diagonal(
    #     x1
    # )[0])
    #
    # print(np.diagonal(
    #     np.linalg.matrix_power(before_matrix, 2) - x1 * x1
    # )[0])
    #
    #
    # print(np.diagonal(
    #     np.linalg.matrix_power(before_matrix, 3) - ((x2 - (x1 * x1)) * x1) - (x2 * x1)
    # )[0])

    def func(k):
        return np.linalg.matrix_power(matrix, k) - \
               sum([func(num) * np.linalg.matrix_power(matrix, k - num) for num in range(1, k)])

    return np.diagonal(func(k))[i - 1]


i, k = 8, 9
print("Задание 6: ", task6(matrix=P, k=k, i=i))


"""
Task 7
"""


def task7(matrix, k, i):
    out = []

    def func(k):
        res = np.linalg.matrix_power(matrix, k) - sum([func(num) * np.linalg.matrix_power(matrix, k - num) for num in range(1, k)])

        out.append(np.diagonal(res))
        return res

    func(k)
    return sum(out)[i - 1]


i, k = 11, 6
print("Задание 7: ", task7(P, k=k, i=i))


"""
Task 8
"""


def task8(matrix, i):
    result = []

    def func(k):
        res = np.linalg.matrix_power(matrix, k) - sum(
            [func(num) * np.linalg.matrix_power(matrix, k - num) for num in range(1, k)])
        result.append(k * np.diagonal(res))
        return res

    func(i)
    # print(result)
    return sum(result)[i-1]


i = 10
print("Задание 8: ", task8(P, i))


"""
Task 9
"""


def task9(matrix):
    L_transpose = np.transpose(matrix)

    D = np.zeros((len(matrix), len(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                D[i][j] = sum(matrix[i])

    M = L_transpose - D

    B = np.zeros(len(matrix))
    B[-1] = 1

    M_ = M
    M_[-1] = 1

    X = np.linalg.matrix_power(M_, -1) @ B
    return X


print("Задание 9: ",task9(matrix=P))
