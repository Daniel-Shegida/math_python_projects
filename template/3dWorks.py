# his import registers the 3D projection
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

L = open("DYNAMICS3_ver2_RES.txt", "wt")

print("ДВИЖЕНИЕ ТЕЛА В СРЕДЕ С СОПРОТИВЛЕНИЕМ")
print("ДВИЖЕНИЕ ТЕЛА В СРЕДЕ С СОПРОТИВЛЕНИЕМ", file=L)

x0 = 0.0 # m
y0 = 0.0 # m
z0 = 0.0 # m
al = 50.0 # grad
al = al*np.pi/180.0 # rad
V0 = 500.0 # m/sec
Vx0 = V0*np.cos(al) # m/sec
Vy0 = 0.0 # m/sec
Vz0 = V0*np.sin(al) # m/sec
m = 0.009 # kg
g = 9.8 # m/sec^2
A = 1.e-5 # N*sec/m
B = 1.e-8 # N*sec^3/m^3
Fwind = 0.01 # N (force of cross wind along y-axis)
tm = 110.0 # sec


def Frv(V):
    global A, B
    # inus because of resistance force
    # in the opposite direction of velocity
    return -(A*V + B*V**3)/V


def system(f, t):
    global m, g, A, B, Fwind
    x = f[0]
    Vx = f[1]
    y = f[2]
    Vy = f[3]
    z = f[4]
    Vz = f[5]
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    dxdt = Vx
    dVxdt = Frv(V)*Vx/m
    dydt = Vy
    dVydt = Fwind/m + Frv(V)*Vy/m
    dzdt = Vz
    dVzdt = -g + Frv(V)*Vz/m
    return [dxdt, dVxdt, dydt, dVydt, dzdt, dVzdt]

nt = 1000
t = np.linspace(0., tm, nt)
sol = odeint(system, [x0, Vx0, y0, Vy0, z0, Vz0], t)
x = sol[:, 0]
Vx = sol[:, 1]
y = sol[:, 2]
Vy = sol[:, 3]
z = sol[:, 4]
Vz = sol[:, 5]

# Simple calculation of Tflight
for i in range(len(z)):
    if z[i] < 0.0:
        Tflight = (t[i]+t[i-1])/2.0
        numnode = i
        print("Node of landing:", numnode)
        print("Tflight=", Tflight)
        print("Node of landing:", numnode, file=L)
        print("Tflight=", Tflight, file=L)
        break
tmax =round(Tflight+0.5)
print("tmax=", tmax)
print("t[numnode]=", t[numnode])
print("x[numnode]=", x[numnode])
print("y[numnode]=", y[numnode])
print("z[numnode]=", z[numnode])
print("Vx[numnode]=", Vx[numnode])
print("Vy[numnode]=", Vy[numnode])
print("Vz[numnode]=", Vz[numnode])

delta = 20.5
x_max = round(x[:numnode].max()+delta)
print("x_max=", x_max)
print("x_max=", x_max, file=L)
Vx_max = round(Vx[:numnode].max()+delta)
print("Vx_max=", Vx_max)
print("Vx_max=", Vx_max, file=L)
y_max = round(y[:numnode].max()+delta)
print("y_max=", y_max)
print("y_max=", y_max, file=L)
Vy_max = round(Vy[:numnode].max()+delta)
print("Vy_max=", Vy_max)
print("Vy_max=", Vy_max, file=L)
z_max = round(z[:numnode].max()+delta)
print("z_max=", z_max)
print("z_max=", z_max, file=L)
Vz_max = round(Vz[:numnode].max()+delta)
print("Vz_max=", Vz_max)
print("Vz_max=", Vz_max, file=L)
plt.plot(t, Vx, 'r-', linewidth=3)
plt.plot(t, [0.0]*nt, 'g-', linewidth=1)
plt.plot([Tflight], [Vx[numnode]], 'go')
plt.axis([0, tmax, 0., Vx_max])
plt.grid(True)
plt.xlabel("t, с")
plt.ylabel("Vx(t), м/с")
plt.savefig("Vx.pdf", dpi=300)
plt.show()
plt.close()
plt.plot(t, x, 'b-', linewidth=3)
plt.plot([Tflight], [x[numnode]], 'go')
plt.axis([0, tmax, 0., x_max])
plt.grid(True)
plt.xlabel("t, с")
plt.ylabel("x(t), м")
plt.savefig("x.pdf", dpi=300)
plt.show()
plt.close()
plt.plot(t, Vy, 'r-', linewidth=3)
plt.plot(t, [0.0]*nt, 'g-', linewidth=1)
plt.plot([Tflight], [Vy[numnode]], 'go')
plt.axis([0, tmax, 0., Vy_max])
plt.grid(True)
plt.xlabel("t, с")
plt.ylabel("Vy(t), м/с")
plt.savefig("Vy.pdf", dpi=300)
plt.show()
plt.close()
plt.plot(t, y, 'b-', linewidth=3)
plt.plot([Tflight], [y[numnode]], 'go')
plt.axis([0, tmax, 0., y_max])
plt.grid(True)
plt.xlabel("t, с")
plt.ylabel("y(t), м")
plt.savefig("y.pdf", dpi=300)
plt.show()
plt.close()
plt.plot(t, Vz, 'r-', linewidth=3)
plt.plot(t, [0.0]*nt, 'g-', linewidth=1)
plt.plot([Tflight], [Vz[numnode]], 'go')
plt.axis([0, tmax, -250., Vz_max])
plt.grid(True)
plt.xlabel("t, с")
plt.ylabel("Vz(t), м/с")
plt.savefig("Vz.pdf", dpi=300)
plt.show()
plt.close()
plt.plot(t, z, 'b-', linewidth=3)
plt.plot([Tflight], [z[numnode]], 'go')
plt.axis([0, tmax, 0., z_max])
plt.grid(True)
plt.xlabel("t, с")
plt.ylabel("z(t), м")
plt.savefig("z.pdf", dpi=300)
plt.show()
plt.close()
xx = x[:numnode]
yy = y[:numnode]
zz = z[:numnode]
print("len(xx)=", len(xx))
plt.plot(xx, zz, 'orangered', linewidth=5)
plt.axis([0, x_max, 0., z_max])
plt.grid(True)
plt.title("Trajectory (XZ)")
plt.xlabel("x, м")
plt.ylabel("z, м")
plt.savefig("trajectory_XZ.pdf", dpi=300)
plt.show()
plt.close()
plt.plot(xx, yy, 'orangered', linewidth=5)
plt.axis([0, x_max, 0., y_max])
plt.grid(True)
plt.title("Trajectory (XY)")
plt.xlabel("x, м")
plt.ylabel("y, м")
plt.savefig("trajectory_XY.pdf", dpi=300)
plt.show()
plt.close()
plt.plot(yy, zz, 'orangered', linewidth=5)
plt.axis([0, y_max, 0., z_max])
plt.grid(True)
plt.title("Trajectory (YZ)")
plt.xlabel("y, м")
plt.ylabel("z, м")
plt.savefig("trajectory_YZ.pdf", dpi=300)
plt.show()
plt.close()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
"""
fig - переменная, содержащая объект figure;
fig.gca () возвращает оси, связанные с figure.
Если для gca задан аргумент projection = …, возвращаются оси,
помеченные указанным тегом (который обычно является строкой)
"""
ax.plot(xx, yy, zz, 'orangered', linewidth=5,
label="trajectory")
ax.legend()
ax.set_xlabel('X, м', fontsize=10)
ax.set_ylabel('Y, м', fontsize=10)
ax.set_zlabel("Z, м", fontsize=10)
#ax.yaxis._axinfo['label']['space_factor'] = 3.0
plt.savefig("trajectory_3d.pdf", dpi=300)
plt.show()