"""
2 П.Г.Доля. Введение в научный Python, с. 240
3 dy(t)/dt = -5*y(t)*t^3;
4 y(-1.5) = 0.01.
5 t in [-1.5, +.1.5].
 / dy1(x)/dx = +y2(x);
{
 \ dy2(x)/dx = -y1(x);
6 """
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

L = open("shegida-odeint34.txt", "w")


def show_analitic_numerical_compare(starting_x, ending_x, analitic_fun, numerical_x, numerical_y, index):
 analitic_x = np.linspace(starting_x, ending_x, 201)
 analitic_y = analitic_fun(analitic_x)

 # create plot
 # plt.figure(1, figsize = (12,8) )
 # plt.plot(xx,yy1,color="blue",linestyle="solid",linewidth=5,label='analytic')

 plt.figure("Закройте окно, чтобы перейти к следующему графику")
 plt.plot(analitic_x, analitic_y, 'b-', linewidth=5, label='analytic')
 plt.plot(numerical_x, numerical_y, 'ro', label="numeric")
 plt.xlabel('X')
 plt.ylabel('Y' + index.__str__() + '(X)')
 if index == 1:
  plt.legend(loc='upper right')
 else:
  plt.legend(loc='upper center')
 plt.axhline(color='gray', zorder=-1)
 plt.axvline(color='gray', zorder=-1)
 plt.grid(True)

 plt.gcf().set_size_inches(10, 6)
 # save plot to file
 plt.savefig('y' + index.__str__() + ' .pdf', dpi=300)
 plt.savefig('y' + index.__str__() + '.eps', dpi=300)
 plt.show()
def dydt(y, t):
 y1, y2 = y
 return y2, -y1

a = 0.0
b = 2.0*np.pi

t = np.linspace(a,b,51)
print("type(t)=", type(t))
y0 = [0,1]
y1, y2 = odeint(dydt, y0, t).T
print("type(y)=", type(y2))
print("type(y)=", type(y2), file=L)
print("y=", y2)
print("y=", y2, file=L)
y = y2.flatten()
print("y=", y2)
print("y=", y2, file=L)
show_analitic_numerical_compare(a, b, np.sin, t, y1, 1)
show_analitic_numerical_compare(a, b, np.cos, t, y2, 2)

