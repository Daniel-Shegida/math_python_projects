# Пакет (библиотека) numpy. Назначение, общая характеристика, примеры
# использования.
import numpy as np

# 17. Пакет (библиотека) numpy. Генерация массивов. Функции array, arange,
# zeros, ones, zeros_like, ones_like, identity, eye.
a = np.array([1, 4, 5, 8], float)

# The arange function is similar to the range function but returns an array instead of a list:
np.arange(1, 6, 2, dtype=int)

# The functions zeros and ones create new arrays of specified dimensions filled with these
# values. These are perhaps the most commonly used functions to create new arrays:
np.ones((2,3), dtype=float)
np.zeros(7, dtype=int)

# The zeros_like and ones_like functions create a new array with the same dimensions and
# type of an existing one:
a = np.array([[1, 2, 3], [4, 5, 6]], float)
np.zeros_like(a)
np.ones_like(a)

# There are also a number of functions for creating special matrices (2D arrays). To create an
# identity matrix of a given size,
# array([[ 1., 0., 0., 0.],
#  [ 0., 1., 0., 0.],
#  [ 0., 0., 1., 0.],
#  [ 0., 0., 0., 1.]])
np.identity(4, dtype=float)

# The eye function returns matrices with ones along the kth diagonal:
#     array([[0., 1., 0., 0.],
#            [0., 0., 1., 0.],
#            [0., 0., 0., 1.],
#            [0., 0., 0., 0.]])
np.eye(4, k=1, dtype=float)


# 18. Пакет (библиотека) numpy. Методы shape, reshape, dtype, copy, tolist, tostring.
# The shape property of an array returns a tuple with the size of each array dimension:
# a = np.array([[1, 2, 3], [4, 5, 6]], float)
a.shape
# (2, 3)

# Arrays can be reshaped using tuples that specify new dimensions. In the following example, we
# turn a ten-element one-dimensional array into a two-dimensional one whose first axis has five
# elements and whose second axis has two elements:
a = np.array(range(10), float)
a = a.reshape((5, 2))

# The dtype property tells you what type of values are stored by the array:
a.dtype

# Keep in mind that Python's name-binding approach still applies to arrays. The copy function can
# be used to create a separate copy of an array in memory if needed:
a = np.array([1, 2, 3], float)
b = a
c = a.copy()

# Lists can also be created from arrays:
a = np.array([1, 2, 3], float)
a.tolist()

# One can convert the raw data in an array to a binary string (i.e., not in human-readable form)
# using the tostring function. The fromstring function then allows an array to be created
# from this data later on. These routines are sometimes convenient for saving large amount of
# array data in binary files that can be read later on:
a2 = np.array([1, 2, 3], float)
s = a2.tostring()

# 19. Пакет (библиотека) numpy. Методы fill, transpose, flatten, concatenate.
# One can fill an array with a single value:
a = np.array([1, 2, 3], float)
a.fill(0)

# Two or more arrays can be concatenated together using the concatenate function with a
# tuple of the arrays to be joined:
a = np.array([1,2], float)
b = np.array([3,4,5,6], float)
c = np.array([7,8,9], float)
np.concatenate((a, b, c))

# Transposed versions of arrays can also be generated, which will create a new array with the final
# two axes switched:
a = np.array(range(6), float).reshape((2, 3))
a.transpose()

# One-dimensional versions of multi-dimensional arrays can be generated with flatten:
a = np.array([[1, 2, 3], [4, 5, 6]], float)
a.flatten()

# 20. Пакет (библиотека) numpy. Применение стандартных математических
# операций к массивам.
# When standard mathematical operations are used with arrays, they are applied on an elementby-element basis. This means that the arrays should be the same size during addition,
# subtraction, etc.:
a = np.array([1,2,3], float)
b = np.array([5,2,6], float)
a + b
a - b
a * b
b / a
a % b
b**a

# For two-dimensional arrays, multiplication remains elementwise and does not correspond to
# matrix multiplication.
a = np.array([[1,2], [3,4]], float)
b = np.array([[2,0], [1,3]], float)
a * b
# array([[2., 0.], [3., 12.]])
# 21. Пакет (библиотека) numpy. Применение функций sqrt, floor, ceil, rint к
# массивам.
# In addition to the standard operators, NumPy offers a large library of common mathematical
# functions that can be applied elementwise to arrays. Among these are the functions: abs,
# sign, sqrt, log, log10, exp, sin, cos, tan, arcsin, arccos, arctan,
# sinh, cosh, tanh, arcsinh, arccosh, and arctanh.
a = np.array([1, 4, 9], float)
np.sqrt(a)
# The functions floor, ceil, and rint give the lower, upper, or nearest (rounded) integer:
a = np.array([1.1, 1.5, 1.9], float)
# вниз
np.floor(a)
# up
np.ceil(a)
np.rint(a)




# 22. Пакет (библиотека) numpy. Основные операции, применяемые к массивам
# (sum, prod, mean, var, std, min, max, argmin, argmax).
# Many functions exist for extracting whole-array properties. The items in an array can be summed
# or multiplied:
a = np.array([2, 4, 3], float)
a.sum()
a.prod()

# A number of routines enable computation of statistical quantities in array datasets, such as the
# mean (average), variance, and standard deviation:
a = np.array([2, 1, 9], float)
a.mean()
# 4.0

# var = x = abs(a - a.mean())**2.
a.var()
# 12.666666666666666
# std = sqrt(var)
a.std()
# 3.5590260840104371

# It's also possible to find the minimum and maximum element values:
a = np.array([2, 1, 9], float)
a.min()
a.max()

# The argmin and argmax functions return the array indices of the minimum and maximum
# values:
a = np.array([2, 1, 9], float)
# 1
a.argmin()
# 2
a.argmax()


# 23. Пакет (библиотека) numpy. Основные операции, применяемые к массивам
# (sorted, sort, clip, unique, diagonal).
# Like lists, arrays can be sorted
a = np.array([6, 2, 5, -1, 0], float)
a.sort()
a
# array([-1., 0., 2., 5., 6.])

# Values in an array can be "clipped" to be within a prespecified range. This is the same as applying
# min(max(x, minval), maxval) to each element x in an array
a = np.array([6, 2, 5, -1, 0], float)
a.clip(0, 5)
# array([ 5., 2., 5., 0., 0.])

# Unique elements can be extracted from an array:
a = np.array([1, 1, 4, 5, 5, 5, 7], float)
np.unique(a)
# array([ 1., 4., 5., 7.])

# For two dimensional arrays, the diagonal can be extracted:
a = np.array([[1, 2], [3, 4]], float)
a.diagonal()
# array([ 1., 4.])



# 24. Пакет (библиотека) numpy. Логические операции, применяемые к массивам
# (включая функции: any, all, logical_and, logical_or, logical_not, where, nonzero,
# isnan, isfinite).
# The any and all operators can be used to determine whether or not any or all elements of a
# Boolean array are true:
a = np.array([1, 3, 0], float)
any(a > 2)
# True
all(a > 2)
# False

# Compound Boolean expressions can be applied to arrays on an element-by-element basis using
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




# 25. Пакет (библиотека) numpy. Применение логических фильтров к массивам.
# Функции take и put.
a = np.array([[6, 4], [5, 9]], float)
a >= 6
# array([[ True, False],
#  [False, True]], dtype=bool)

# A special function take is also available to perform selection with integer arrays. This works in
# an identical manner as bracket selection
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int)
a.take(b)
# array([ 2., 2., 4., 8., 6., 4.])
# функция put заменяет элементы в массиве а по индексам получаемые из первого массива элементами, получаемыми из 2 массива
# в нашем случа из b
a = np.array([0, 1, 2, 3, 4, 5], float)
b = np.array([9, 8, 7], float)
a.put([0, 3], b)
a
# array([ 9., 1., 2., 8., 4., 5.])

# Note that the value 7 from the source array b is not used, since only two indices [0, 3] are
# specified. The source array will be repeated as necessary if not the same size:
a = np.array([0, 1, 2, 3, 4, 5], float)
a.put([0, 3], 5)
a
# array([ 5., 1., 2., 5., 4., 5.])



# 26. Пакет (библиотека) numpy. Операции с векторами и матрицами (dot, inner,
# outer, cross, det, inv).
# NumPy provides many functions for performing standard vector and matrix multiplication
# routines. To perform a dot product, (a1 * b1 + a2 * b2...)
a = np.array([1, 2, 3], float)
b = np.array([0, 1, 1], float)
np.dot(a, b)

# The dot function also generalizes to matrix multiplication:
a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
c = np.array([[1, 1], [4, 0]], float)
np.dot(b, a)
# array([ 6., 11.])


# It is also possible to generate inner, outer, and cross products of matrices and vectors. For
# vectors, note that the inner product is equivalent to the dot product:
a = np.array([1, 4, 0], float)
b = np.array([2, 2, 1], float)
# Given two vectors, a = [a0, a1, ..., aM] and b = [b0, b1, ..., bN], the outer product [1] is:
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
# Перекрестное произведение двух векторов дает вектор, который перпендикулярен плоскости, образованной входными векторами, и его величина пропорциональна площади, охватываемой параллелограммом, образованным этими входными векторами.
# Источник: https://tonais.ru/library/funktsiya-numpy-cross-v-python
np.cross(a, b)
# array([ 4., -1., -6.])
# cross(A,B) = [(4*1 - 0*4), -(1*1-0*2), (1*2-4*2)] = [ 4., -1., -6.]
# в двоичной системе только (1*2-4*2)

# NumPy also comes with a number of built-in routines for linear algebra calculations. These can
# be found in the sub-module linalg. Among these are routines for dealing with matrices and
# their inverses. The determinant of a matrix can be found:
a = np.array([[4, 2, 0], [9, 3, 7], [1, 2, 1]], float)
np.linalg.det(a)

# The inverse of a matrix can be found:
b = np.linalg.inv(a)


# 27. Пакет (библиотека) numpy. Функции для работы с полиномами (poly, roots,
# polyint, polyder, polyval, polyfit)
# NumPy supplies methods for working with polynomials. Given a set of roots, it is possible to
# show the polynomial coefficients:
np.poly([-1, 1, 1, 10])
# array([ 1, -11, 9, 11, -10])
# Here, the return array gives the coefficients corresponding to x^4 - 11x^3 + 9x^2 + 11x - 10

# The opposite operation can be performed: given a set of coefficients, the root function returns
# all of the polynomial roots:
# x^3 + 4x^2 -2x + 3
np.roots([1, 4, -2, 3])
# array([-4.57974010+0.j , 0.28987005+0.75566815j,
#  0.28987005-0.75566815j])

# Coefficient arrays of polynomials can be integrated. for example:
# x^3 + x^2 + x + 1 -> x^4\4 + x^3\3 + x^2\2 + x + c
#  c in default is zerro
np.polyint([1, 1, 1, 1])
# array([ 0.25 , 0.33333333, 0.5 , 1. , 0. ])

# Similarly, derivatives can be taken:
np.polyder([1./4., 1./3., 1./2., 1., 0.])
# array([ 1., 1., 1., 1.])


# The function polyval evaluates a polynomial at a particular point.
# то есть просто подстановка чисел и счет
np.polyval([1, -2, 0, 2], 4)
# 34


# Finally, the polyfit function can be used to fit a polynomial of specified order to a set of data
# using a least-squares approach:
# или если выразиться проще, то оно по точкам в 2 мерном пространстве сощданным по массивам х и у пытается найти полином
#  что проходит сквозь эти точки
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 2, 1, 3, 7, 10, 11, 19]
np.polyfit(x, y, 2)
# array([ 0.375 , -0.88690476, 1.05357143])


