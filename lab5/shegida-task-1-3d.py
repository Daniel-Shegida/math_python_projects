"""""
 Шегида Даниил Леонидович 1 курс магистратуры
Задание №1 по теме "Комп. мод. движения тела в среде с сопротивлением""
"""

import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt

L = open("task1.txt", "wt")

print("Шегида  1 курс магистратуры\nЗадание №1 по теме Комп. мод. движения тела в среде с сопротивлением\n")
print("Шегида  1 курс магистратуры\nЗадание №1 по теме Комп. мод. движения тела в среде с сопротивлением\n", file=L)

x0 = 0.0 # м
y0 = 0.0 # м
z0 = 0.0 # м
al = 60.0 # град
al = al*np.pi/180.0 # рад
V0 = 75.0 # м/с
Vx0 = V0 * np.cos(al) # м/с
Vy0 = 0.0 # м/с
Vz0 = V0*np.sin(al) # м/с
m = 0.009 # кг
g = 9.8 # м/с^2
A = 1.e-5 # Н*с/м
B = 1.e-8 # Н*с^3/м^3
# A = 555 # Н*с/м
# B = 333 # Н*с^3/м^3
tm = 110.0 # с

def Fwind():
    return 0.5

def FResist(V):
    return -(A * V + B * V ** 3) / V

def dfdt(f, t):
    x, Vx, y, Vy, z, Vz = f
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    dxdt = Vx
    # dVxdt = 0
    dVxdt = FResist(V) * Vx / m
    dydt = Vy
    dVydt = Fwind() / m + FResist(V) * Vy / m
    dzdt = Vz
    # dVzdt = -g
    dVzdt = -g + FResist(V) * Vz / m
    return [dxdt, dVxdt, dydt, dVydt, dzdt, dVzdt]

def dfdt2(f, t):
    x, Vx, y, Vy, z, Vz = f
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    dxdt = Vx
    dVxdt = 0
    # dVxdt = FResist(V) * Vx / m
    dydt = Vy
    # dVydt = 0.01 / m + FResist(V) * Vy / m
    dVydt = 0
    dzdt = Vz
    dVzdt = -g
    # dVzdt = -g + FResist(V) * Vz / m
    return [dxdt, dVxdt, dydt, dVydt, dzdt, dVzdt]

nt = 1000
t = np.linspace(0., tm, nt)
sol = odeint(dfdt, [x0, Vx0, y0, Vy0, z0, Vz0], t)
x = sol[:, 0]
Vx = sol[:, 1]
y = sol[:, 2]
Vy = sol[:, 3]
z = sol[:, 4]
Vz = sol[:, 5]
sol2 = odeint(dfdt2, [x0, Vx0, y0, Vy0, z0, Vz0], t)
x2 = sol2[:, 0]
Vx2 = sol2[:, 1]
y2 = sol2[:, 2]
Vy2 = sol2[:, 3]
z2 = sol2[:, 4]
Vz2 = sol2[:, 5]


arrayCount = 0
tFlight = 0
arrayCount2 = 0
tFlight2 = 0
for count in range(len(z)):
    if z[count] < 0.0 and (tFlight == 0):
        tFlight = abs((t[count] + t[count-1]) / 2.0)
        arrayCount = count

for count in range(len(z2)):
    if z2[count] < 0.0 and (tFlight2 == 0):
        tFlight2 = abs((t[count] + t[count-1]) / 2.0)
        arrayCount2 = count


print("Используя алгоритм, вычислим приблизительные значения времени полёта в среде и вакууме:")
print("Используя алгоритм, вычислим приблизительные значения времени полёта в среде и вакууме:", file=L)

print("Время полёта в среде: \ntFlight =", tFlight)
print("Время полёта в среде: \ntFlight =", tFlight, file=L)
print("Время полёта в вакууме : \ntFlight =", tFlight2)
print("Время полёта в вакууме: \ntFlight =", tFlight2, file=L)

arrayCount2 = 0
tFlight2 = 0
for count in range(len(z2)):
    if z2[count] < 0.0 and (tFlight2 == 0):
        tFlight2 = abs((t[count] + t[count-1]) / 2.0)
        arrayCount2 = count



zMax = round(z[:arrayCount].max())
zMax2 = round(z2[:arrayCount].max())

print("\nМаксимальная высота подъёма в среде: \nz_max =", zMax)
print("\nМаксимальная высота подъёма в среде: \nz_max =", zMax, file=L)
print("\nМаксимальная высота подъёма в вакууме: \nz_max =", zMax)
print("\nМаксимальная высота подъёма в вакууме: \nz_max =", zMax, file=L)


plt.plot(t, Vx, 'r-', linewidth=3, label="Траектория в эксперементальной среде ")
plt.plot(t, Vx2, 'b--', linewidth=2, label="Траектория в вакууме")
plt.plot([tFlight], [Vx[arrayCount]], 'ko')
plt.grid(True)
plt.axis([0, t[arrayCount] + 5, 0., Vx[:arrayCount].max() + 50])
plt.title("Vx(t)")
plt.xlabel("t, с")
plt.ylabel("Vx(t), м/с")
plt.savefig("Vx.pdf", dpi=300)

plt.show()
plt.close()

plt.plot(t, x, 'b-', linewidth=3, label="Траектория в эксперементальной среде ")
plt.plot(t, x2, 'r--', linewidth=2, label="Траектория в вакууме")
plt.plot([tFlight], [x[arrayCount]], 'ko')
plt.grid(True)
plt.axis([0, t[arrayCount] + 5, 0., x[:arrayCount].max() + 5000])
plt.title("x(t)")
plt.xlabel("t, с")
plt.ylabel("x(t), м")
plt.savefig("x.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, Vy, 'r-', linewidth=3, label="Траектория в эксперементальной среде ")
plt.plot(t, Vy2, 'b--', linewidth=2, label="Траектория в вакууме")
plt.plot([tFlight], [Vy[arrayCount]], 'go')
plt.grid(True)
plt.title("Vy(t)")
plt.xlabel("t, с")
plt.ylabel("Vy(t), м/с")
plt.savefig("Vy.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, y, 'b-', linewidth=3, label="Траектория в эксперементальной среде ")
plt.plot(t, y2, 'r--', linewidth=2, label="Траектория в вакууме")
plt.plot([tFlight], [y[arrayCount]], 'go')
plt.grid(True)
plt.title("y(t)")
plt.xlabel("t, с")
plt.ylabel("y(t), м")
plt.savefig("y.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, Vz, 'r-', linewidth=3, label="Траектория в эксперементальной среде ")
plt.plot(t, Vz2, 'b--', linewidth=2, label="Траектория в вакууме ")
plt.plot([tFlight], [Vz[arrayCount]], 'ko')
plt.grid(True)
plt.axis([0, t[arrayCount] + 5, Vz[:arrayCount].min() - 100, Vz[:arrayCount].max()])
plt.title("Vz(t)")
plt.xlabel("t, с")
plt.ylabel("Vz(t), м/с")
plt.savefig("Vz.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, z, 'b-', linewidth=3, label="Траектория в эксперементальной среде ")
plt.plot(t, z2, 'r--', linewidth=3, label="Траектория в вакууме")
plt.plot([tFlight], [z[arrayCount]], 'ko')
plt.grid(True)
plt.axis([0, t[arrayCount] + 5, -1000, z[:arrayCount].max() + 500])
plt.title("z(t)")
plt.xlabel("t, с")
plt.ylabel("z(t), м")
plt.savefig("z.pdf", dpi=300)
plt.show()
plt.close()

xx = x[:arrayCount]
yy = y[:arrayCount]
zz = z[:arrayCount]

xx2 = x2[:arrayCount]
yy2 = y2[:arrayCount]
zz2 = z2[:arrayCount]

plt.plot(xx, yy, 'orangered', linewidth=5, label="Траектория в эксперементальной среде ")
plt.plot(xx2, yy2, 'b--', linewidth=3, label="Траектория в вакууме")
plt.grid(True)
plt.title("График траектории")
plt.xlabel("x, м")
plt.ylabel("y, м")
plt.savefig("trajectory_XY.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(xx, zz, 'orangered', linewidth=5, label="Траектория в эксперементальной среде ")
plt.plot(xx2, zz2, 'b--', linewidth=3, label="Траектория в вакууме")
plt.grid(True)
plt.title("График траектории")
plt.xlabel("x, м")
plt.ylabel("z, м")
plt.savefig("trajectory_XZ.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(yy, zz, 'orangered', linewidth=5, label="Траектория в эксперементальной среде ")
plt.plot(yy2, zz2, 'b--', linewidth=3, label="Траектория в вакууме")
plt.grid(True)
plt.title("График траектории")
plt.xlabel("y, м")
plt.ylabel("z, м")
plt.savefig("trajectory_YZ.pdf", dpi=300)
plt.show()
plt.close()


fig = plt.figure()

ax = fig.add_subplot(projection = "3d")
ax.plot(xx, yy, zz, 'orangered', linewidth=5, label="Траектория в эксперементальной среде ")
ax.plot(xx2, yy2, zz2, 'b--', linewidth=3, label="Траектория в вакууме")
ax.legend()
plt.title("График траектории 3D")
ax.set_xlabel('X, м', fontsize=10)
ax.set_ylabel('Y, м', fontsize=10)
ax.set_zlabel("Z, м", fontsize=10)
plt.savefig("trajectory_3d.pdf", dpi=300)
plt.show()