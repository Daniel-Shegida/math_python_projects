import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

'''
Тема Обыкновенные диф уравнения и систеимы
Источник ВВедение в научный мпитон Доля П.Г 239с
Студент Шегида ДЛ
Группа ктмфо
x'' + x^3 = sin(t), x(0) = x'(0) = 0
'''
L = open("shegida-odeint12.txt", "w")
def f(y, t):
 y1, y2 = y # вводим имена искомых функций
 return [y2,-y1**3+np.sin(t)]

t = np.linspace(0,50,201)
print("type(t)=", type(t))
y0 = [0, 0]
[y1,y2]= odeint(f, y0, t, full_output=False).T
print("type(y)=", type(y2))
print("type(y)=", type(y2), file=L)
print("y=", y2)
print("y=", y2, file=L)
y = y2.flatten()
# Строим график решения.
fig = plt.figure(facecolor='white')
plt.plot(t,y1, '-o',linewidth=2) # график решения
plt.grid(True)
plt.show()