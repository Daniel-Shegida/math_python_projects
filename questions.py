import numpy as np
import sys

label = """
Функции пакета numpy

Выполнил: Шегида Даниил Леонидович
"""

class DoubleWrite:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, s):
        self.file1.write(s)
        self.file2.write(s)

    def flush(self):
        self.file1.flush()
        self.file2.flush()

logfile = open('numpy.txt', 'w')
sys.stdout = DoubleWrite(sys.stdout, logfile)
print(label)

# Пакет (библиотека) numpy. Назначение, общая характеристика, примеры
# использования.


# 17. Пакет (библиотека) numpy. Генерация массивов. Функции array, arange,
# zeros, ones, zeros_like, ones_like, identity, eye.
# Создание массива из элементов 1 4 5 8 с типом float
a = np.array([1, 4, 5, 8], float)
print('Создание массива из элементов 1 4 5 8 с типом float:')
print('a = np.array([1, 4, 5, 8], float)')
print(a)


# создание массива с использованием синтаксиса range
a = np.arange(1, 6, 2, dtype=int)
print('создание массива с использованием синтаксиса range:')
print('np.arange(1, 6, 2, dtype=int):')
print(a)

# возвращяет массив нулей или единиц с указанной размерностью
a = np.ones((2,3), dtype=float)
b = np.zeros(7, dtype=int)
print('возвращяет массив нулей или единиц с указанной размерностью:')
print('np.ones((2,3), dtype=float)')
print(a)
print('np.zeros(7, dtype=int')
print(b)

# Возвращает массив единиц, с размерностью выбранного массива
a = np.array([[1, 2, 3], [4, 5, 6]], float)
b = np.zeros_like(a)
c = np.ones_like(a)
print('Возвращает массив единиц, с размерностью выбранного массива:')
print('np.zeros_like(a):')
print(b)
print('np.ones_like(a):')
print(c)

# Возвращяет единичный массив
# array([[ 1., 0., 0., 0.],
#  [ 0., 1., 0., 0.],
#  [ 0., 0., 1., 0.],
#  [ 0., 0., 0., 1.]])
a = np.identity(4, dtype=float)
print('Возвращяет единичный массив:')
print('np.identity(4, dtype=float):')
print(a)

# Взвращяет единичный массив с двигом к:
#     array([[0., 1., 0., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 0., 1.],
#            [0., 0., 0., 0.]])
a = np.eye(4, k=1, dtype=float)
print('Взвращяет единичный массив с двигом к:')
print('np.eye(4, k=1, dtype=float):')
print(a)


# 18. Пакет (библиотека) numpy. Методы shape, reshape, dtype, copy, tolist, tostring.
# возвращение размерности массива
a = np.array([[1, 2, 3], [4, 5, 6]], float)
b = a.shape
# (2, 3)
print('возвращение размерности массива:')
print('a = np.array([[1, 2, 3], [4, 5, 6]], float)')
print('a.shape)')
print(b)

# Переделывает массив в выбранную размерность
a = np.array(range(10), float)
b = a.reshape((5, 2))
print('Переделывает массив в выбранную размерность:')
print('np.array(range(10), float)')
print('a.reshape((5, 2))')
print(b)

# Возвращяет тип хранимых переменных в массиве:
b = a.dtype
print('Возвращяет тип хранимых переменных в массиве:')
print('a = np.array(range(10), float)')
print('a.dtype)')
print(b)

# Возвращяет копию массива:
a = np.array([1, 2, 3], float)
b = a
c = a.copy()
a[0] = 0
b
c
print('Возвращяет копию массива:')
print('a = np.array([1, 2, 3], float)')
print('b = a')
print('c = a.copy()')
print('a[0] = 0')
print(b)
print(c)

# Полиморф массива в лист:
a = np.array([1, 2, 3], float)
b = a.tolist()
print('Полиморф массива в лист:')
print('a = np.array([1, 2, 3], float)')
print('a.tolist()')
print(b)

# Полиморф массива в бинарную строку:
a = np.array([1, 2, 3], float)
s = a.tostring()
print('Полиморф массива в бинарную строкут:')
print('a = np.array([1, 2, 3], float)')
print('s = a.tostring()')
print(s)

# 19. Пакет (библиотека) numpy. Методы fill, transpose, flatten, concatenate.
# Наполняет массив следующей переменной:
a = np.array([1, 2, 3], float)
a.fill(0)
print('Наполняет массив следующей переменной::')
print('a = np.array([1, 2, 3], float)')
print('a.fill(0)')
print(a)

# Два или более массива могут быть соединены вместе:
a = np.array([1,2], float)
b = np.array([3,4,5,6], float)
c = np.array([7,8,9], float)
d = np.concatenate((a, b, c))
print(' Два или более массива могут быть соединены вместе:')
print('a = np.array([1,2], float)')
print('b = np.array([3,4,5,6], float)')
print('c = np.array([7,8,9], float)')
print('np.concatenate((a, b, c))')
print(d)

# Команда транспонирования двухмерных массивов:
a = np.array(range(6), float).reshape((2, 3))
print('Команда транспонирования двухмерных массивов:')
print('a = np.array(range(6), float).reshape((2, 3))')
print('a.transpose()')
print(a)
a.transpose()
print(a)

# Возвращяет 1-мерный массив из многомерного:
a = np.array([[1, 2, 3], [4, 5, 6]], float)
a.flatten()
print('Возвращяет 1-мерный массив из многомерного:')
print('a = np.array([[1, 2, 3], [4, 5, 6]], float)')
print('a.flatten()')
print(a)


# 20. Пакет (библиотека) numpy. Применение стандартных математических
# операций к массивам.
# Все математические операции применяются по элементам друг к другу:
a = np.array([1,2,3], float)
b = np.array([5,2,6], float)
a + b
a - b
a * b
b / a
a % b
b**a
print('Все математические операции применяются по элементам друг к другу:')
print('a = np.array([1,2,3], float)')
print('b = np.array([5,2,6], float)')
print('a + b')
print(a + b)

# Двухмерные массивы также умножаются по элементам
a = np.array([[1,2], [3,4]], float)
b = np.array([[2,0], [1,3]], float)
a * b
# array([[2., 0.], [3., 12.]])
print('Двухмерные массивы также умножаются по элементам:')
print('a = np.array([[1,2], [3,4]], float)')
print('b = np.array([[2,0], [1,3]], float)')
print('a * b')
print(a * b)

# 21. Пакет (библиотека) numpy. Применение функций sqrt, floor, ceil, rint к
# массивам.
# Библиотека поддерживает моножество функций, такие как abs,
# sign, sqrt, log, log10, exp, sin, cos, tan, arcsin, arccos, arctan,
# sinh, cosh, tanh, arcsinh, arccosh, and arctanh.
a = np.array([1, 4, 9], float)
b = np.sqrt(a)
print('# Библиотека поддерживает моножество функций, такие как abs, sign, sqrt, log, log10, exp, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, and arctanh.')
print('a = np.array([1, 4, 9], float)')
print('np.sqrt(a)')
print(b)

# Функции floor, ceil, and rint округляют вниз, вверх или ближайшее:
a = np.array([1.1, 1.5, 1.9], float)
# вниз
b = np.floor(a)
# up
c = np.ceil(a)
d = np.rint(a)
print(' Функции floor, ceil, and rint округляют вниз, вверх или ближайшее:')
print('a = np.array([1.1, 1.5, 1.9], float)')
print('np.floor(a)')
print(b)
print('np.ceil(a)')
print(c)
print('np.rint(a)')
print(d)

# 22. Пакет (библиотека) numpy. Основные операции, применяемые к массивам
# (sum, prod, mean, var, std, min, max, argmin, argmax).
# Возвращяет сумму или умножение всех элементов массива:
a = np.array([2, 4, 3], float)
b = a.sum()
c = a.prod()
print(' Возвращяет сумму или умножение всех элементов массива')
print('a = np.array([2, 4, 3], float)')
print('a.sum()')
print(b)
print('a.prod()')
print(c)

# Возвращяет среднее массива:
a = np.array([2, 1, 9], float)
b = a.mean()
# 4.0
print(' Возвращяет среднее массива:')
print('a = np.array([2, 1, 9], float)')
print('a.mean()')
print(b)


#возвращяет квадратичное отклонение
# var = x = abs(a - a.mean())**2.
b = a.var()
# 12.666666666666666
print(' возвращяет квадратичное отклонение:')
print('a = np.array([2, 1, 9], float)')
print('a.var()')
print(b)

#  Возвращяет отклонение
# std = sqrt(var)
b = a.std()
# 3.5590260840104371
print(' Возвращяет отклонение:')
print('a = np.array([2, 1, 9], float)')
print('a.std()')
print(b)

# Возвращение максимального или минимального элемента:
a = np.array([2, 1, 9], float)
b = a.min()
c = a.max()
print(' ВВозвращение максимального или минимального элемента:')
print('a = np.array([2, 1, 9], float)')
print('a.min()')
print(b)
print('a.max()')
print(c)

# Возврашение индекса максимального или минимального элемента массива:
a = np.array([2, 1, 9], float)
# 1
b = a.argmin()
# 2
c = a.argmax()
print(' Возврашение индекса максимального или минимального элемента массива:')
print('a = np.array([2, 1, 9], float)')
print('a.argmin()')
print(b)
print('a.argmax()')
print(c)

# 23. Пакет (библиотека) numpy. Основные операции, применяемые к массивам
# (sorted, sort, clip, unique, diagonal).
# Сортировка массиовов
a = np.array([6, 2, 5, -1, 0], float)
a.sort()

# array([-1., 0., 2., 5., 6.])
print(' Сортировка массиовов:')
print('a = np.array([6, 2, 5, -1, 0], float)')
print('a.sort()')
print(a.sort())


# Возможность выставлять максимальные или минимальные величины в массивах
a = np.array([6, 2, 5, -1, 0], float)
a.clip(0, 5)
# array([ 5., 2., 5., 0., 0.])
print('Возможность выставлять максимальные или минимальные величины в массивах')
print('a = np.array([6, 2, 5, -1, 0], float)')
print('a.clip(0, 5)')
print(a.clip(0, 5))

# Возвращяет уникальные элементы массива:
a = np.array([1, 1, 4, 5, 5, 5, 7], float)
np.unique(a)
# array([ 1., 4., 5., 7.])
print('Возвращяет уникальные элементы массива:')
print('a = np.array([1, 1, 4, 5, 5, 5, 7], float)')
print('np.unique(a)')
print(np.unique(a))

# Возвращяет знааачение по диагонали:
a = np.array([[1, 2], [3, 4]], float)
a.diagonal()
# array([ 1., 4.])
print('Возвращяет знааачение по диагонали:')
print('a = np.array([[1, 2], [3, 4]], float)')
print('a.diagonal()')
print(a.diagonal())


# 24. Пакет (библиотека) numpy. Логические операции, применяемые к массивам
# (включая функции: any, all, logical_and, logical_or, logical_not, where, nonzero,
# isnan, isfinite).
# Можно поставить логическое выражение для любого или для всех элементов:
a = np.array([1, 3, 0], float)
any(a > 2)
# True
all(a > 2)
# False
print('Можно поставить логическое выражение для любого или для всех элементов:')
print('a = np.array([1, 3, 0], float)')
print('any(a > 2)')
print(any(a > 2))
print('all(a > 2)')
print(all(a > 2))

# Более сложные булевы выражения можно сделать с помошью команд
# special functions logical_and, logical_or, and logical_not.
a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3)
# y([ True, False, False], dtype=bool)
b = np.array([True, False, True], bool)
np.logical_not(b)
# y([False, True, False], dtype=bool)
c = np.array([False, True, False], bool)
np.logical_or(b, c)
# array([ True, True, True], dtype=bool)
print('# Более сложные булевы выражения можно сделать с помошью команд special functions logical_and, logical_or, and logical_not.')
print('a = np.array([1, 3, 0], float)')
print('np.logical_and(a > 0, a < 3)')
print(np.logical_and(a > 0, a < 3))
print('b = np.array([True, False, True], bool)')
print('np.logical_not(b)')
print(np.logical_not(b))
print('c = np.array([False, True, False], bool)')
print('np.logical_or(b, c)')
print(np.logical_or(b, c))


# 25. Пакет (библиотека) numpy. Применение логических фильтров к массивам.
# Функции take и put.
a = np.array([[6, 4], [5, 9]], float)
a >= 6
# array([[ True, False],
#  [False, True]], dtype=bool)
print('Применение логических фильтров к массивам..')
print('a = np.array([[6, 4], [5, 9]], float)')
print('a >= 6')
print(a >= 6)


# take берет из массива к которому принимаается элементы по индексам, которые указываются в переменной
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int)
a.take(b)
# array([ 2., 2., 4., 8., 6., 4.])
print('take берет из массива к которому принимаается элементы по индексам, которые указываются в переменной')
print('a = np.array([2, 4, 6, 8], float)')
print('b = np.array([0, 0, 1, 3, 2, 1], int)')
print('a.take(b)')
print(a.take(b))


# функция put заменяет элементы в массиве а по индексам получаемые из первого массива элементами, получаемыми из 2 массива
# в нашем случае из b
a = np.array([0, 1, 2, 3, 4, 5], float)
b = np.array([9, 8, 7], float)
a.put([0, 3], b)
# array([ 9., 1., 2., 8., 4., 5.])
print('функция put заменяет элементы в массиве а по индексам получаемые из первого массива элементами, получаемыми из 2 массивав нашем случае из b')
print('a = np.array([0, 1, 2, 3, 4, 5], float)')
print('b = np.array([9, 8, 7], float)')
print('a.put([0, 3], b)')
print(a.put([0, 3], b))

#Если значение массива недостаточно, то он продлевается:
a = np.array([0, 1, 2, 3, 4, 5], float)
a.put([0, 3], 5)

# array([ 5., 1., 2., 5., 4., 5.])



# 26. Пакет (библиотека) numpy. Операции с векторами и матрицами (dot, inner,
# outer, cross, det, inv).
# Для нахождения определителя используется дот (a1 * b1 + a2 * b2...)
a = np.array([1, 2, 3], float)
b = np.array([0, 1, 1], float)
np.dot(a, b)
print('Для нахождения определителя используется дот (a1 * b1 + a2 * b2...)')
print('a = np.array([1, 2, 3], float)')
print('b = np.array([0, 1, 1], float)')
print('np.dot(a, b)')
print(np.dot(a, b))

#  dot также применим для произведения  матриц:
a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
c = np.array([[1, 1], [4, 0]], float)
np.dot(b, a)
# array([ 6., 11.])
print('dot также применим для произведения  матриц:')
print('a = np.array([[0, 1], [2, 3]], float)')
print('b = np.array([2, 3], float)')
print('c = np.array([[1, 1], [4, 0]], float)')
print('np.dot(b, a)')
print(np.dot(b, a))

# Для более привычных произведений массивов, можно использовать inner, outer, and cross :
a = np.array([1, 4, 0], float)
b = np.array([2, 2, 1], float)
# для двух векторов a = [a0, a1, ..., aM] и b = [b0, b1, ..., bN], outer product [1] is:
# [[a0*b0  a0*b1 ... a0*bN ]
#  [a1*b0    .
#  [ ...          .
#  [aM*b0            aM*bN ]]
np.outer(a, b)
# array([[ 2., 2., 1.],
#  [ 8., 8., 4.],
#  [ 0., 0., 0.]])
np.inner(a, b)
# 10.0
# Перекрестное произведение двух векторов дает вектор, который перпендикулярен плоскости, образованной входными
# векторами, и его величина пропорциональна площади, охватываемой параллелограммом, образованным этими входными векторами.
np.cross(a, b)
# array([ 4., -1., -6.])
# cross(A,B) = [(4*1 - 0*4), -(1*1-0*2), (1*2-4*2)] = [ 4., -1., -6.]
# в двоичной системе только (1*2-4*2)
print('Для более привычных произведений массивов, можно использовать inner, outer, and cross:')
print('a = np.array([1, 4, 0], float)')
print('b = np.array([2, 2, 1], float)')
print('np.outer(a, b)')
print(np.outer(a, b))
print('np.inner(a, b)')
print(np.inner(a, b))
print('np.cross(a, b)')
print(np.cross(a, b))

# Для нахождения определителя, нужно воспользоваться сабмудлем linalg. командой det:
a = np.array([[4, 2, 0], [9, 3, 7], [1, 2, 1]], float)
np.linalg.det(a)
print('Для нахождения определителя, нужно воспользоваться сабмудлем linalg. командой det')
print('a = np.array([[4, 2, 0], [9, 3, 7], [1, 2, 1]], float)')
print('np.linalg.det(a)')
print(np.linalg.det(a))

# инверсия матрицы можно найти :
b = np.linalg.inv(a)
print('инверсия матрицы можно найти')
print('a = np.array([[4, 2, 0], [9, 3, 7], [1, 2, 1]], float)')
print('b = np.linalg.inv(a)')
print(b)

# 27. Пакет (библиотека) numpy. Функции для работы с полиномами (poly, roots,
# polyint, polyder, polyval, polyfit)
# с помощью NumPy можно получить кэфициэнты, введя коэфициенты уравнения s:
np.poly([-1, 1, 1, 10])
# array([ 1, -11, 9, 11, -10])
# что аналогично x^4 - 11x^3 + 9x^2 + 11x - 10

print('с помощью NumPy можно получить кэфициэнты, введя коэфициенты уравнения s')
print('np.poly([-1, 1, 1, 10])')
print(np.poly([-1, 1, 1, 10]))

# Обратная ситуация, можно получить корни введя коэфициэнты  :
# x^3 + 4x^2 -2x + 3
np.roots([1, 4, -2, 3])
# array([-4.57974010+0.j , 0.28987005+0.75566815j,
#  0.28987005-0.75566815j])
print('Обратная ситуация, можно получить корни введя коэфициэнты ')
print('np.roots([1, 4, -2, 3])')
print(np.roots([1, 4, -2, 3]))

# коэфиэнты массивов могут быть интегрированы
# Coefficient arrays of polynomials can be integrated. for example:
# x^3 + x^2 + x + 1 -> x^4\4 + x^3\3 + x^2\2 + x + c
#  c in default is zerro
np.polyint([1, 1, 1, 1])
# array([ 0.25 , 0.33333333, 0.5 , 1. , 0. ])
print('коэфиэнты массивов могут быть интегрированы ')
print('np.polyint([1, 1, 1, 1])')
print(np.polyint([1, 1, 1, 1]))

# и наоборот, можно получить производную:
np.polyder([1./4., 1./3., 1./2., 1., 0.])
# array([ 1., 1., 1., 1.])
print('и наоборот, можно получить производную ')
print('np.polyder([1./4., 1./3., 1./2., 1., 0.])')
print(np.polyder([1./4., 1./3., 1./2., 1., 0.]))

# В функцию можно подставить число с
np.polyval([1, -2, 0, 2], 4)
# 34
print('В функцию можно подставить число с ')
print('np.polyval([1, -2, 0, 2], 4)')
print(np.polyval([1, -2, 0, 2], 4))


# Можно, если выразиться проще, то оно по точкам в 2 мерном пространстве сощданным по массивам х и у пытается найти полином
#  что проходит сквозь эти точки
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 2, 1, 3, 7, 10, 11, 19]
np.polyfit(x, y, 2)
# array([ 0.375 , -0.88690476, 1.05357143])
print('Можно, если выразиться проще, то оно по точкам в 2 мерном пространстве сощданным по массивам х и у пытается найти полином  что проходит сквозь эти точки')
print('x = [1, 2, 3, 4, 5, 6, 7, 8]')
print('y = [0, 2, 1, 3, 7, 10, 11, 19]')
print('np.polyfit(x, y, 2)')
print(np.polyfit(x, y, 2))

