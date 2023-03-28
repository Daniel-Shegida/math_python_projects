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
fig, ax = plt.subplots(9, 1, figsize=(15, 20))
ax1 = ax[0]
ax2 = ax[1]
# ax2_right = ax2.twinx()
ax3 = ax[2]
# ax3_right = ax3.twinx()
ax4 = ax[3]
ax5 = ax[4]
ax6 = ax[5]
ax7 = ax[6]
ax8 = ax[7]
ax9 = ax[8]
fig2 = plt.figure()
a2x2 = fig2.add_subplot(projection='3d')



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

ax1.plot(t, [0.0]*nt, 'g-', linewidth=1)
ax1.plot([Tflight], [Vx[numnode]], 'go')
ax1.axis([0, tmax, 0., Vx_max])
ax1.grid(True)
# ax1.xlabel("t, с")
# ax1.ylabel("Vx(t), м/с")
# ax1.savefig("Vx.pdf", dpi=300)

ax2.plot([Tflight], [x[numnode]], 'go')
ax2.axis([0, tmax, 0., x_max])
ax2.grid(True)
# ax2.xlabel("t, с")
# ax2.ylabel("x(t), м")
# ax2.savefig("x.pdf", dpi=300)
ax3.plot(t, [0.0]*nt, 'g-', linewidth=1)
ax3.plot([Tflight], [Vy[numnode]], 'go')
ax3.axis([0, tmax, 0., Vy_max])
ax3.grid(True)
# ax3.xlabel("t, с")
# ax3.ylabel("Vy(t), м/с")
# ax3.savefig("Vy.pdf", dpi=300)

ax4.plot([Tflight], [y[numnode]], 'go')
ax4.axis([0, tmax, 0., y_max])
ax4.grid(True)
# ax4.xlabel("t, с")
# ax4.ylabel("y(t), м")
# ax4.savefig("y.pdf", dpi=300)

ax5.plot(t, [0.0]*nt, 'g-', linewidth=1)
ax5.plot([Tflight], [Vz[numnode]], 'go')
ax5.axis([0, tmax, -250., Vz_max])
ax5.grid(True)
# ax5.xlabel("t, с")
# ax5.ylabel("Vz(t), м/с")
# ax5.savefig("Vz.pdf", dpi=300)

ax6.plot([Tflight], [z[numnode]], 'go')
ax6.axis([0, tmax, 0., z_max])
ax6.grid(True)
# ax6.xlabel("t, с")
# ax6.ylabel("z(t), м")
# ax6.savefig("z.pdf", dpi=300)
# plt.show()
# plt.close()
xx = x[:numnode]
yy = y[:numnode]
zz = z[:numnode]
print("len(xx)=", len(xx))
ax7.plot(xx, zz, 'orangered', linewidth=5)
ax7.axis([0, x_max, 0., z_max])
ax7.grid(True)
# ax7.title("Trajectory (XZ)")
# ax7.xlabel("x, м")
# ax7.ylabel("z, м")
# ax7.savefig("trajectory_XZ.pdf", dpi=300)
# plt.show()
# plt.close()
ax8.axis([0, x_max, 0., y_max])
ax8.grid(True)
# ax8.title("Trajectory (XY)")
# ax8.xlabel("x, м")
# ax8.ylabel("y, м")
# plt.savefig("trajectory_XY.pdf", dpi=300)
# plt.show()
# plt.close()
ax9.axis([0, y_max, 0., z_max])
ax9.grid(True)
# ax9.title("Trajectory (YZ)")
# ax9.xlabel("y, м")
# ax9.ylabel("z, м")
# plt.savefig("trajectory_YZ.pdf", dpi=300)
# plt.show()
# plt.close()
#

"""
fig - переменная, содержащая объект figure;
fig.gca () возвращает оси, связанные с figure.
Если для gca задан аргумент projection = …, возвращаются оси,
помеченные указанным тегом (который обычно является строкой)
"""

a2x2.legend()
a2x2.set_xlabel('X, м', fontsize=10)
a2x2.set_ylabel('Y, м', fontsize=10)
a2x2.set_zlabel("Z, м", fontsize=10)
#ax.yaxis._axinfo['label']['space_factor'] = 3.0



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
A = 0 # N*sec/m
B = 0 # N*sec^3/m^3
Fwind = 0.0 # N (force of cross wind along y-axis)
tm = 110.0 # sec
fig, ax = plt.subplots(9, 1, figsize=(15, 20))
ax1 = ax[0]
ax2 = ax[1]
# ax2_right = ax2.twinx()
ax3 = ax[2]
# ax3_right = ax3.twinx()
ax4 = ax[3]
ax5 = ax[4]
ax6 = ax[5]
ax7 = ax[6]
ax8 = ax[7]
ax9 = ax[8]
fig2 = plt.figure()
a2x2 = fig2.add_subplot(projection='3d')


t = np.linspace(0., tm, nt)
sol = odeint(system, [x0, Vx0, y0, Vy0, z0, Vz0], t)
x = sol[:, 0]
Vx = sol[:, 1]
y = sol[:, 2]
Vy = sol[:, 3]
z = sol[:, 4]
Vz = sol[:, 5]


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


ax1.plot(t, Vx, 'b-', linewidth=3)
ax2.plot(t, x, 'b-', linewidth=3)
ax3.plot(t, Vy, 'r-', linewidth=3)
ax4.plot(t, y, 'b-', linewidth=3)
ax5.plot(t, Vz, 'r-', linewidth=3)
ax6.plot(t, z, 'b-', linewidth=3)
ax7.plot(xx, zz, 'orangered', linewidth=5)
ax8.plot(xx, yy, 'orangered', linewidth=5)
ax9.plot(yy, zz, 'orangered', linewidth=5)

a2x2.plot(xx, yy, zz, 'orangered', linewidth=5,
label="trajectory")

fig2.savefig("trajectory_3d.pdf", dpi=300)
# plt.show()
fig.savefig('result_1.pdf', bbox_inches='tight')
