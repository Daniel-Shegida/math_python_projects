import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve


def fun_opt(par):
    global xfin0, zfin0
    al = par[0]
    V0 = par[1]
    xz = xz_point(m1, g1, A1,B1, tm1, nt1, x01, z01, al, V0)
    return [(xz[0] - xfin1) ** 2, (xz[1] - zfin1) ** 2]
def fSolveFun_opt(par,xfin0, zfin0, m,g,A,B, tm, nt, x0, z0,):
    global xfin1, zfin1,m1,g1,A1,B1,tm1,nt1,x01,z01, al1, V01
    xfin1 = xfin0
    zfin1 = zfin0
    m1 = m
    g1 = g
    A1 = A
    B1 = B
    tm1 = tm
    nt1 = nt
    x01 = x0
    z01 = z0


    return fsolve(fun_opt, par)


def xz_point(m, g, A, B, tm, nt, x0, z0, al, V0):
    global m3, g3, A3, B3, tm2, nt2, x02, z02
    tm2 = tm
    nt2 = nt
    x02 = x0
    z02 = z0
    m3 = m
    g3 = g
    A3 = A
    B3 = B

    Vx0 = V0 * np.cos(al)
    Vz0 = V0 * np.sin(al)
    t = np.linspace(0., tm, nt)
    sol = odeintSystem(m, g, A, B, [x0, Vx0, z0, Vz0], t)
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

def odeintSystem(m, g, A, B, f,  t):

    global  m2, g2, A2, B2
    m2 = m
    g2 = g
    A2 = A
    B2 = B
    return odeint(system, f, t)

def Frv(V, A, B):
    return -(A * V + B * V ** 3) / V
def system(f, t):
    # global m, g, A, B

    x = f[0]
    Vx = f[1]
    z = f[2]
    Vz = f[3]
    V = np.sqrt(Vx ** 2 + Vz ** 2)
    dxdt = Vx
    dVxdt = Frv(V, A2, B2) * Vx / m2
    dzdt = Vz
    dVzdt = -g2 + Frv(V, A2, B2) * Vz / m2
    return [dxdt, dVxdt, dzdt, dVzdt]