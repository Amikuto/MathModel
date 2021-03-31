import numpy as np


def create_matrix(L, m, u, n):
    """
    Создание матрицы по заданным параметрам

    Args:
        L (int): интенсивность поступления λ
        m (int): кол-во каналов обслуживания m
        u (int): интенсивность обслуживания μ
        n (int): максимальный размер очереди
    """
    p_matrix = np.zeros((m + n + 1, m + n + 1))
    for i in range(m + n):
        p_matrix[i, i + 1] = L
        if i < m:
            p_matrix[i + 1, i] = (u * (i + 1))
        else:
            p_matrix[i + 1, i] = u * m
    return p_matrix


def task_a(matrix):
    """
    Установившиеся вероятности
    """
    diag_res = []
    for i in range(matrix.shape[0]):
        diag_res.append(matrix[i, :].sum())

    D = np.diag(diag_res)
    M = matrix.transpose() - D
    # print(M)
    # print()
    # print()
    M_ = np.copy(M)
    M_[-1, :] = 1
    # print(M_)

    b_vector = np.zeros(M_.shape[0])
    b_vector[-1] = 1
    X = np.linalg.inv(M_).dot(b_vector)
    return X


def task_b(vector):
    """
    Вероятность отказа в обслуживании
    """
    return vector[-1]


def task_c(vector, L):
    """
    Относительная и абсолютная интенсивность обслуживания
    """
    relative = 1 - vector[-1]
    return relative, relative * L


def task_d(vector, m, n):
    """
    Средняя длина очереди
    """
    s = 0
    for i in range(1, n + 1):
        s += i * vector[m + i]
    return s


def task_e(vector, m, u, n):
    """
    Среднее время в очереди
    """
    s = 0
    for i in range(n):
        s += ((i + 1) / (m * u) * vector[m + i])
    return s


def task_f(vector, m, n):
    """
    Среднее число занятых каналов
    """
    s = 0
    for i in range(1, m + n + 1):
        if i <= m:
            s += i * vector[i]
        else:
            s += m * vector[i]
    return s


def task_g(vector, m):
    """
    Вероятность не ждать в очереди
    """
    return sum(vector[:m])


def task_h(matrix):
    """
    Среднее время простоя системы массового обслуживания
    """
    return 1 / np.sum(matrix, -1)


L = 31
m = 3
u = 16
n = 16
matrix = create_matrix(L, m, u, n)
print("Матрица переходов:")
print(matrix)


vector = task_a(matrix)
print(f"Составьте граф марковского процесса, запишите систему уравнений Колмогорова и \
найдите установившиеся вероятности состояний:\n--> {vector}")


answer_b = task_b(vector)
print(f"Найдите вероятность отказа в обслуживании:\n--> {answer_b}")


relative, absolute = task_c(vector, L)
print("Найдите относительную и абсолютную интенсивность обслуживания:")
print(f"Относительная: {relative}\nАбсолютная: {absolute}")


answer_d = task_d(vector, m, n)
print(f"Найдите среднюю длину в очереди:\n--> {answer_d}")


answer_e = task_e(vector, m, u, n)
print(f"Найдите среднее время в очереди:\n--> {answer_e}")


answer_f = task_f(vector, m, n)
print(f"Найдите среднее число занятых каналов:\n--> {answer_f}")


answer_g = task_g(vector, m)
print(f"Найдите вероятность того, что поступающая заявка не будет ждать в очереди:\n--> {answer_g}")


answer_h = task_h(matrix)
print(f"Найти среднее время простоя системы массового обслуживания:\n--> {answer_h[0]}")

import numpy as np
from functools import lru_cache


def task1(matrix, k):
    """
    Вероятность перехода из состояния i в j за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    return np.linalg.matrix_power(matrix, k)


def task2(matrix, a_0, k):
    """
    Вероятность состояния за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        a_0 (numpy.ndarray): вероятности состояние в начальный момент времениъ
        k (int): кол-во шагов
    """
    return a_0.dot(np.linalg.matrix_power(matrix, k))


def task3(matrix, k):
    """
    Вероятность первого перехода за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    matrix_p = np.copy(matrix)
    for _ in range(1, k):
        matrix_p = skip_j_state(matrix, matrix_p)
    return matrix_p


def skip_j_state(matrix, matrix_pr):
    """
    Пропуск j состояния

    Args:
        matrix (numpy.ndarray): матрица переходов
        matrix_pr (numpy.ndarray): матрица переходов предыдущая
    """
    len_p = len(matrix)
    new_matrix = np.zeros((len_p, len_p))
    for i in range(len_p):
        for j in range(len_p):
            s = 0
            for m in range(len_p):
                if m != j:
                    s += matrix[i, m] * matrix_pr[m, j]
            new_matrix[i, j] = s
    return new_matrix


def task4(matrix, k):
    """
    Вероятность перехода не позднее чем за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    matrix_pr, result = np.copy(matrix), np.copy(matrix)
    for _ in range(1, k):
        matrix_pr = skip_j_state(matrix, matrix_pr)
        result += matrix_pr
    return result


def task5(matrix):
    """
    Среднее количество шагов для перехода из состояния i в j

    Args:
        matrix (numpy.ndarray): матрица переходов
    """
    matrix_pr, result = np.copy(matrix), np.copy(matrix)
    for g in range(1, 1500):
        matrix_pr = skip_j_state(matrix, matrix_pr)
        result += g * matrix_pr
    return result


def task6(matrix, k):
    """
    Вероятность первого возвращения на k-ом шаге

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    matrix_pr = np.copy(matrix)

    @lru_cache(maxsize=None)
    def f_jj(k):
        return np.linalg.matrix_power(matrix_pr, k) - sum(
            [f_jj(m) * np.linalg.matrix_power(matrix_pr, k - m) for m in range(1, k)])

    return np.diagonal(f_jj(k))


def task7(matrix, k):
    out = []

    @lru_cache(maxsize=None)
    def func(k):
        res = np.linalg.matrix_power(matrix, k) - sum([func(num) * np.linalg.matrix_power(matrix, k - num) for num in range(1, k)])

        out.append(np.diagonal(res))
        return res

    func(k)
    return sum(out)


def task8(matrix):
    """
    Среднее время возвращения

    Args:
        matrix (numpy.ndarray): матрица переходов
    """
    matrix_pr, result = np.copy(matrix), []

    @lru_cache(maxsize=None)
    def f_jj(k=500):
        res = np.linalg.matrix_power(matrix_pr, k) - sum(
            [f_jj(m) * np.linalg.matrix_power(matrix_pr, k - m) for m in range(1, k)])
        result.append(k * np.diagonal(res))
        return res

    f_jj()
    return sum(result)


def task9(matrix):
    """
    Установившиеся вероятности

    Args:
        matrix (numpy.ndarray): матрица переходов
    """
    matrix_ = np.copy(matrix).transpose()
    np.fill_diagonal(matrix_, np.diagonal(matrix_) - 1)
    matrix_[-1, :] = 1

    b_vector = np.zeros(len(matrix))
    b_vector[-1] = 1
    X = np.linalg.inv(matrix_).dot(b_vector)
    return X


matrix = np.array([
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

k, i, j = 10, 1, 7
answer1 = task1(matrix, k)
print(f"Вероятность того, что за k={k} шагов система перейдет из состояния {i} в состояние {j} \n--> {answer1[i-1][j-1]}")

k = 10
a_0 = np.array([0.04, 0.04, 0.09, 0.03, 0.11, 0, 0.19, 0.17, 0.12, 0.16, 0.05])
answer2 = task2(matrix, a_0, k)
print(f"Вероятности состояний системы спустя k={k} шагов, если в начальный "
      f"момент вероятность состояний были следующими \nA={a_0}\n\nОтвет: {answer2}")


k, i, j = 8, 8, 7
answer3 = task3(matrix, k)
print(f"Вероятность первого перехода за k={k} шагов из состояния {i} в состояние {j} \n--> {answer3[i-1][j-1]}")


i, j, k = 6, 8, 5
answer4 = task4(matrix, k)
print(f"Вероятность перехода из состояния {i} в состояние {j} не позднее чем за k={k} шагов \n--> {answer4[i-1][j-1]}")


i, j = 2, 8
answer5 = task5(matrix)
print(f"Среднее количество шагов для перехода из состояния {i} в состояние {j} \n--> {answer5[i-1][j-1]}")


i, k = 8, 9
answer6 = task6(matrix, k)
print(f"Вероятность первого возвращения в состояние {i} за k={k} шагов\n--> {answer6[i-1]}")

i, k = 11, 6
answer7 = task7(matrix, k)
print(f"Вероятность возвращения в состояние {i} не позднее чем за k={k} шагов\n--> {answer7[i-1]}")

i = 10
answer8 = task8(matrix)
print(f"Среднее время возвращения в состояние {i} \n--> {answer8[i-1]}")

answer9 = task9(matrix)
print(f"Установившиеся вероятности:\n--> {answer9}")