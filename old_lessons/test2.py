## решаем задачу коши z" + 1\5 z' + z = 0 z(0) = 0, z'(0) = 1
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def f(y, t):
    y1, y2 = y  # вводим имена искомых функций
    return [y2, -0.2 * y2 - y1]


# Решаем систему ОДУ.
t = np.linspace(0, 20, 41)
y0 = [0, 1]
w = odeint(f, y0, t)
w
y1 = w[:, 0]  # вектор значений решения
y2 = w[:, 1]  # вектор значений производной
# Строим график решения (значения искомой функции находятся в первом
# столбце матрицы w, т.е. в векторе y1)
fig = plt.figure(facecolor='white')  # следующий рисунок слева
plt.plot(t, y1, '-o', linewidth=2)
plt.ylabel("z")
plt.xlabel("t")
plt.grid(True)
# Строим график фазовой траектории. Последовательность решения та же, но
# количество искомых точек возьмем больше.
t = np.linspace(0, 20, 201)
y0 = [0, 1]
[y1, y2] = odeint(f, y0, t, full_output=False).T
fig = plt.figure(facecolor='white')  # предыдущий рисунок справа
plt.plot(y1, y2, linewidth=2)
plt.grid(True)
