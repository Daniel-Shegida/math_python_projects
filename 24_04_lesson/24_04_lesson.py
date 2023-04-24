'''
z'' + z = 0
z(0) - 0
z(pi/2) = 1

transform into
y1'(t) = y2(t)
y2'(t) = -y1(t)
y1(0) = 0, y1([pi/2) = 1
'''

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

def plot_solution(ax, x, exact_sol, sol_x, sol_y, xlabel="$t$"):
    ax.plot(x, exact_sol(x), color="green", label="Точное решение")
    ax.scatter(sol_x, sol_y, color="red", marker=".", label="Приближенное решение")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$y$")
    ax.legend()
    plt.show()
def exact_solution(t):
    return np.sin(t)

def f(t, y):
    return [
        y[1],
        -y[0]
        ]
def boundary_residual(ya, yb):
    return np.array([
        ya[0] - 0,
        yb[0] - 1
        ])


a, b = 0, np.pi/2
N = 30
x = np.linspace(a, b, N)
y_guess = np.zeros((2, N), dtype=float)

sol = integrate.solve_bvp(f, boundary_residual, x, y_guess)

fig, ax = plt.subplots(figsize=(8, 8), layout="tight")
plot_solution(ax, x, exact_solution, sol.x, sol.y[0])