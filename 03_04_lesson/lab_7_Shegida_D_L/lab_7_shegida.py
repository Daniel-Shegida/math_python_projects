label = """
Шегида Даниил Леонидович 1 курс магистратуры
Обратная задача (Комп. мод. движения тела в среде с сопротивлением)
 """
import sys

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import interpolate
from scipy import optimize


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

logfile = open('lab_7_shegida.py.txt', 'w')
sys.stdout = DoubleWrite(sys.stdout, logfile)
print(label)

m = 0.009  # kg
g = 9.8  # m/sec^2
A = 1.e-5  # N*sec/m
B = 1.e-8  # N*sec^3/m^3
x0 = 0.0  # m
z0 = 0.0  # m
tm = 60.0  # sec
print('начальные значения:')
print('масса m :', m)
print('ускорение свободного падения g :', g)
print('коэффициенты среды A и B :', A, B)


def fun_opt(par):
    global xfin0, zfin0
    al = par[0]
    V0 = par[1]
    xz = xz_point(al, V0)
    return [(xz[0] - xfin0) ** 2, (xz[1] - zfin0) ** 2]


def Frv(V):
    global A, B
    return -(A * V + B * V ** 3) / V


def system(f, t):
    global m, g, A, B

    x = f[0]
    Vx = f[1]
    z = f[2]
    Vz = f[3]
    V = np.sqrt(Vx ** 2 + Vz ** 2)
    dxdt = Vx
    dVxdt = Frv(V) * Vx / m
    dzdt = Vz
    dVzdt = -g + Frv(V) * Vz / m
    return [dxdt, dVxdt, dzdt, dVzdt]


def xz_point(al, V0):
    global tm, nt, x0, z0

    Vx0 = V0 * np.cos(al)
    Vz0 = V0 * np.sin(al)
    t = np.linspace(0., tm, nt)
    sol = odeint(system, [x0, Vx0, z0, Vz0], t)
    x = sol[:, 0]
    z = sol[:, 2]
    for i in range(len(z)):
        if z[i] < 0.0:
            Tflight = (t[i] + t[i - 1]) / 2.0
            numnode = i
            break
    xfin = (x[numnode] + x[numnode - 1]) / 2.0
    zfin = (z[numnode] + z[numnode - 1]) / 2.0
    return [xfin, zfin]


nt = 90000

t = np.linspace(0., tm, nt)

al = 60.0  # degrees
al *= np.pi / 180.0  # radians
V0 = 1000.0  # m/sec
res = xz_point(al, V0)

xfin0 = res[0]
zfin0 = res[1]

# print("al=", al / (np.pi / 180))
# print("V0=", V0)
# print("xfin0 =", xfin0)
# print("zfin0 =", zfin0)

al_start = 40.0 * (np.pi / 180)
V0_start = 500.0
root = fsolve(fun_opt, [al_start, V0_start])
# print("root =", root)
al_opt = root[0]
V0_opt = root[1]
print('оптимальный угол для запуска: ', al_opt)
print('оптимальная скорость для запуска: ', V0_opt)


Vx0 = V0_opt * np.cos(al_opt)  # m/sec
Vz0 = V0_opt * np.sin(al_opt)  # m/sec
sol = odeint(system, [x0, Vx0, z0, Vz0], t)
x = sol[:, 0]
Vx = sol[:, 1]
z = sol[:, 2]
Vz = sol[:, 3]

delta = 2.5

x_max = round(x.max() + delta)
print('максимальная дальность по найденным условиям: ', x_max)
z_max = round(z.max() + delta)
print('максимальная высота по найденным условиям: ', z_max)

# Simple calculation of Tflight
for i in range(len(z)):
    if z[i] < 0.0:
        Tflight = (t[i] + t[i - 1]) / 2.0
        numnode = i
        break

tmax = round(Tflight + 0.5)
print("Время полета по найденным условиям=", Tflight)
# print("t[numnode]=", t[numnode])
# print("x[numnode]=", x[numnode])
# print("z[numnode]=", z[numnode])

tN = []
ztN = []
for count in range(numnode - 10, numnode + 10, 1):
    tN.append(t[count])
    ztN.append(z[count])

zInterpoler = interpolate.interp1d(tN, ztN,  kind='quadratic')

tFlightInterpoler1 = optimize.bisect(zInterpoler, tN[0], tN[10])

print("\n время полета найденное через функцию interp1d:")

print("interp1d время полёта: \ntFlightInterpoler1 =", tFlightInterpoler1)


plt.plot(t, x, 'b-', linewidth=3)
plt.plot([Tflight], [xfin0], 'ro')
plt.axis([0, tmax, 0., x_max])
plt.grid(True)
plt.xlabel("t c")
plt.ylabel("x(t) m")
plt.savefig("x.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, z, 'b-', linewidth=3)
plt.plot([Tflight], [zfin0], 'ro')
plt.axis([0, tmax, -30., z_max + 50])
plt.grid(True)
plt.xlabel("t c")
plt.ylabel("z(t) m")
plt.savefig("z.pdf", dpi=300)
plt.show()
plt.close()

xx = x[:numnode]
zz = z[:numnode]

plt.plot(xx, zz, 'orangered', linewidth=5)
plt.plot([xfin0], [zfin0], 'ro')
plt.axis([0, x_max + 50, -30.0, z_max + 50])
plt.grid(True)
plt.title("Trajectory")
plt.xlabel("x m")
plt.ylabel("z m")
plt.savefig("trajectory.pdf", dpi=300)
