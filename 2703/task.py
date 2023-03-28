import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import interpolate

al = 0  # Угол, градусы
al = al * np.pi / 180  # Угол, радианы
x0 = 0  # Начальные точки, м
z0 = 0  #
V0 = 500  # Начальная скорость, m/s
m = 0.009  # Масса, кг
g = 9.8  # Ускорение свободного падения, m /(s^2)
tm = 105  # Время, с
fig, ax = plt.subplots(3, 1, figsize=(15, 20))
ax1 = ax[0]
ax2 = ax[1]
ax2_right = ax2.twinx()
ax3 = ax[2]
ax3_right = ax3.twinx()


A, B = 0, 0
Vx0 = 0  # Начальная скорость по направления
Vz0 = V0
Fside = 0  # Сила, кг*м/с^2

def Frv(v):
    global A, B
    return -(A * v + B * v ** 3) / v


def dydx(y, t):
    x, vx, z, vz = y
    V = np.sqrt(vx ** 2 + vz ** 2)
    return (
        vx,
        (Frv(V) * vx / m) + Fside / m,
        vz,
        (Frv(V) * vz / m) - g,
    )


def calculate(color, label):

    t = np.linspace(0, tm, tm * 1000)
    x, Vx, z, Vz = odeint(dydx, [x0, Vx0, z0, Vz0], t).T

    z_new = z[z >= 0]

    t_interpolate = t[len(z_new) - 50: len(z_new) + 50]
    z_interpolate = z[len(z_new) - 50: len(z_new) + 50]
    f = interpolate.interp1d(t_interpolate, z_interpolate)


    x = x[:len(z)]
    Vx = Vx[:len(z)]
    Vz = Vz[:len(z)]
    t = t[:len(z)]

    ax1.plot(t, np.abs(Vz), color, label=label)

    ax2.plot(t, z, color, label=label)

    ax3.plot(x, z, color, label=label)

    z_max = max(z)
    x_max = x[-1]
    Vz_t0 = t[np.where(np.isclose(Vz, [0], atol=1e-3))[0][0]]
    x_h_max_index, = np.where(z == z_max)
    x_h_max = x[x_h_max_index[0]]

    ax3.plot([x_h_max], [z_max], marker='*', color=color, markersize=10)

    return t[-1], z_max, Vz_t0, x_max, x_h_max


t2_1, h2_1, t1_1, x3_1_max, x3_1_hmax = calculate('r', 'Без всего')
print(h2_1)

A = 10 ** -5  # Параметр внешней среды, N * s/m
B = 10 ** -8  # Параметр внешней среды, N * s^3/m^3

t2_2, h2_2, t1_2, x3_2_max, x3_2_hmax = calculate('b', 'С A и B')

Fside = 0.15  # Сила, кг*м/с^2
t2_3, h2_3, t1_3, x3_3_max, x3_3_hmax = calculate('g', 'С силой, с А и с B')


ax1.legend(loc="upper center", fontsize=15)
ax2.legend(loc="upper left", fontsize=15)
ax3.legend(loc="upper right", fontsize=15)

max_t = max(t2_1, t2_2, t2_3)
max_h = max(h2_1, h2_2, h2_3)
max_x = max(x3_1_max, x3_2_max, x3_3_max)

ax1.set_ylabel('|Vz|, m/s')
ax1.grid(True)
ax2.grid(True)
ax2_right.grid(True)
ax2.set_ylabel('z, m')
ax3.set_xlabel('t, s')
ax3.grid(True)
ax3_right.grid(True)

ax2_xticks = list(ax2.get_xticks()) + [t2_1, t2_2, t2_3]
ax23_yticks = list(ax2.get_yticks())

ax2.set_xticks(ax2_xticks)
ax2_right.set_yticks([h2_2])
ax2_right.get_yticklabels()[0].set_color('blue')
ax2.set_yticks(ax23_yticks + [h2_1, h2_3])
ax2.get_yticklabels()[-1].set_color('green')
ax2.get_yticklabels()[-2].set_color('red')

ax1_xticks = list(ax1.get_xticks()) + [t1_1, t1_2, t1_3]
ax1_xticks.remove(20)
ax1.set_xticks(ax1_xticks)

ax3_xticks = list(ax3.get_xticks()) + [x3_1_max, x3_2_max, x3_3_max, x3_1_hmax, x3_2_hmax, x3_3_hmax]
ax3.set_xticks(ax3_xticks)
ax3.set_yticks(ax23_yticks + [h2_1, h2_2])
ax3.get_yticklabels()[-2].set_color('red')
ax3.get_yticklabels()[-1].set_color('blue')
ax3_right.set_yticks([h2_3])
ax3_right.get_yticklabels()[0].set_color('green')

ax1.tick_params(rotation=45, axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax2.tick_params(rotation=45, axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax2_right.tick_params(axis='y', labelsize=15)
ax3.tick_params(rotation=45, axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3_right.tick_params(axis='y', labelsize=15)

ax1.set_ylim((0, V0 + 10))
ax1.set_xlim((0, max_t + 3))
ax2.set_ylim((0, max_h + 200))
ax2_right.set_ylim((0, max_h + 200))
ax2.set_xlim((0, max_t + 3))
ax3.set_ylim((0, max_h + 200))
ax3.set_xlim((0, max_x + 200))
ax3_right.set_ylim((0, max_h + 200))


fig.savefig('result_1.pdf', bbox_inches='tight')

