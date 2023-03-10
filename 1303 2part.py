"""
2 Start in the XZ plane at an angle of 60 degrees in a
medium with resistance.
3 The OZ axis is perpendicular to the surface.
4 dx(t)/dt = Vx(t)
5 dVx(t)/dt = Frv(V(t))*Vx(t)/m
6 dz(t)/dt = Vz(t)
7 dVz(t)/dt = -g + Frv(V(t))*Vz(t)/m
8 ---
9 V(t) = (Vx(t)^2 + Vy(t)^2)^0.5
10 ---
11 Also see: Movement of a body thrown at an angle to the
horizon in a vacuum
12 https://www.matematicus.ru/en/mechanics-and-physics/movement
-of-a-body-thrown-at-an-angle-to-the-horizon
13 """

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

m = 0.009 # kg
g = 9.8 # m/sec^2
A = 0 # N*sec/m
B = 0 # N*sec^3/m^3
al = 90.0 # degrees
al *= np.pi/180.0 # radians
V0 = 1000.0 # m/sec
x0 = 0.0 # m
z0 = 0.0 # m
Vx0 = V0*np.cos(al) # m/sec
Vz0 = V0*np.sin(al) # m/sec
tm = 2000.0 # sec


def Frv(V):
    global A, B
 # minus because of resistance force
 # in the opposite direction of velocity
    return -(A*V + B*V**3)/V


def system(f, t):
 global m, g, A, B
 x = f[0]
 Vx = f[1]
 z = f[2]
 Vz = f[3]
 V = np.sqrt(Vx**2 + Vz**2)
 dxdt = Vx
 dVxdt = Frv(V)*Vx/m
 dzdt = Vz
 dVzdt = -g + Frv(V)*Vz/m
 return [dxdt, dVxdt, dzdt, dVzdt]

print("al=", al/(np.pi/180))

nt = 2000
t = np.linspace(0., tm, nt)
sol = odeint(system, [x0, Vx0, z0, Vz0], t)
x = sol[:, 0]
Vx = sol[:, 1]
z = sol[:, 2]
Vz = sol[:, 3]

delta = 2.5

x_max = round(x.max()+delta)
print("x_max=", x_max)
Vx_max = round(Vx.max()+delta)
print("Vx_max=", Vx_max)
z_max = round(z.max()+delta)
print("z_max=", z_max)
Vz_max = round(Vz.max()+delta)
print("Vz_max=", Vz_max)

print("len(z)=", len(z))

# Simple calculation of Tflight
for i in range(len(z)):
 if z[i] < 0.0:
    Tflight = (t[i]+t[i-1])/2.0
    numnode = i
    print("Node of landing:", numnode)
    print("Tflight=", Tflight)
    break

tmax =round(Tflight+0.5)
print("tmax=", tmax)
print("t[numnode]=", t[numnode])
print("x[numnode]=", x[numnode])
print("z[numnode]=", z[numnode])
print("Vx[numnode]=", Vx[numnode])
print("Vz[numnode]=", Vz[numnode])

plt.plot(t, Vx, 'r-', linewidth=3)
plt.plot(t, [0.0]*nt, 'g-', linewidth=1)
plt.plot([Tflight], [Vx[numnode]], 'bo')
plt.axis([0, tmax, 0., Vx_max])
plt.grid(True)
plt.xlabel("t")
plt.ylabel("Vx(t)")
plt.savefig("Vx.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, x, 'b-', linewidth=3)
plt.axis([0, tmax, 0., x_max])
plt.grid(True)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("x.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, Vz, 'r-', linewidth=3)
plt.plot(t, [0.0]*nt, 'g-', linewidth=1)
plt.axis([0, tmax, -250., Vz_max])
plt.grid(True)
plt.xlabel("t")
plt.ylabel("Vz(t)")
plt.savefig("Vz.pdf", dpi=300)
plt.show()
plt.close()

plt.plot(t, z, 'b-', linewidth=3)
plt.axis([0, tmax, 0., z_max+50])
plt.grid(True)
plt.xlabel("t")
plt.ylabel("z(t)")
plt.savefig("z.pdf", dpi=300)
plt.show()
plt.close()

xx = x[:numnode]
zz = z[:numnode]
print("len(xx)=", len(xx))

plt.plot(xx, zz, 'orangered', linewidth=5)
plt.axis([0, x_max+50, 0., z_max+50])
plt.grid(True)
plt.title("Trajectory")
plt.xlabel("x")
plt.ylabel("z")
plt.savefig("trajectory.pdf", dpi=300)
plt.show()
plt.close()