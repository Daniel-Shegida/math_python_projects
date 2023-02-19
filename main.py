# Метод Эйлера. Задача Коши
# x: [a,b]

import math
import numpy as np
import matplotlib.pyplot as plt


def f1(x, y1, y2):
    return y2


def f2(x, y1, y2):
    return -y1


def euler2(x_array, y1_starting, y2_starting, steps):
    y1_numeric_values = np.zeros(steps)
    y2_numeric_values = np.zeros(steps)
    # начальные значения
    y1_numeric_values[0] = y1_starting
    y2_numeric_values[0] = y2_starting
    for i in range(1, steps - 1):
        y1_numeric_values[i] = y1_numeric_values[i - 1] + h * f1(x_array[i - 1], y1_numeric_values[i - 1],
                                                                 y2_numeric_values[i - 1])
        y2_numeric_values[i] = y2_numeric_values[i - 1] + h * f2(x_array[i - 1], y1_numeric_values[i - 1],
                                                                 y2_numeric_values[i - 1])
    return y1_numeric_values, y2_numeric_values


def print_analitic_numerical_compare(x_numerical_values, y_numerical_values, analitic_fun):
    #  во столько раз уменьшится срез
    step = 2
    x = x_numerical_values[0:n:step]
    y1 = y_numerical_values[0:n:step]
    nn = len(y1)
    print("len(y1)=", nn)

    sline1 = "=" * 50

    print(sline1)
    str0 = "    x"
    str1 = "      y1(x)"
    str2 = "     sin(x)"
    str3 = "   y1(x)-sin(x)"
    print('{0:8s}{1:14s}{2:14s}{3:14s}'.format(str0, str1, str2, str3))
    print(sline1)
    for i in np.arange(0, nn):
        ares = analitic_fun(x[i])
        dif = y1[i] - ares
        print('{0:8.5f}{1:14.5e}{2:14.5e}{3:8.5f}'.format(x[i], y1[i], ares, dif))


def show_analitic_numerical_compare(starting_x, ending_x, analitic_fun, numerical_x, numerical_y):
    analitic_x = np.linspace(starting_x, ending_x, 201)
    analitic_y = analitic_fun(analitic_x)

    # create plot
    # plt.figure(1, figsize = (12,8) )
    # plt.plot(xx,yy1,color="blue",linestyle="solid",linewidth=5,label='analytic')
    plt.plot(analitic_x, analitic_y, 'b-', linewidth=5, label='analytic')
    plt.plot(numerical_x, numerical_y, 'ro', label="numeric")
    plt.xlabel('X')
    plt.ylabel('Y1(X)')
    plt.legend(loc='upper right')
    plt.axhline(color='gray', zorder=-1)
    plt.axvline(color='gray', zorder=-1)
    plt.grid(True)

    plt.gcf().set_size_inches(10, 6)
    # save plot to file
    plt.savefig('y1.pdf', dpi=300)
    plt.savefig('y1.eps', dpi=300)

    plt.show()


pi = math.pi
# начальные значения задачи
# начальные и конечные значения х
a = 0.0
b = 2.0 * pi
# кол-во узлов
n = 101
# приближение по х
h = float((b - a) / (n - 1))
#  начальное значение y1
y1_starting_value = 0.0
# начальное значение y2
y2_starting_value = 1.0
# массив равномерно размещенных х от начального значения а до конечного б по шагам h
xarray = np.arange(a, b + 0.0001, h)

# print
print("h=", h)
print(len(xarray))
print(xarray[0])

# РЕШЕНИЕ ЗАДАЧИ
# нахождение массива значений y1 и у2 по всем выбранным х
y1array, y2array = euler2(xarray, y1_starting_value, y2_starting_value, n)

print_analitic_numerical_compare(xarray, y1array, np.sin)

show_analitic_numerical_compare(a, b, np.sin, xarray, y1array)
show_analitic_numerical_compare(a, b, np.cos, xarray, y2array)
