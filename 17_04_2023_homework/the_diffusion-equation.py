import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a, b = -5, 5  # size of box
N = 51  # number of subdivisions
x = np.linspace(a, b, N)  # positions of subdivisions
h = x[1] - x[0]  # d i s c r e t i s a t i o n stepsize in x - direction


def total(u):
    """ Computes total number of moles in u . """
    return ((b - a) / float(N) * np.sum(u))


def gaussdistr(mean, sigma, x):
    """ Return gauss distribution for given numpy array x """
    return 1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mean) ** 2 / sigma ** 2)


# starting c on fi gu ra ti on for u (x , t0 )
u = gaussdistr(mean=0., sigma=0.5, x=x)


def compute_g(u, D, h):
    """ given a u (x , t ) in array , compute g (x , t )= D * d ^2 u / dx ^2
    using central differences with spacing h ,
    and return g (x , t ). """
    d2u_dx2 = np.zeros(u.shape, np.float)
    for i in range(1, len(u) - 1):
        d2u_dx2[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / h ** 2
    # special cases at boundary : assume Neuman boundary
    # conditions , i . e . no change of u over boundary
    # so that u [0] - u [ -1]=0 and thus u [ -1]= u [0]
    i = 0
    d2u_dx2[i] = (u[i + 1] - 2 * u[i] + u[i]) / h ** 2
    # same at other end so that u [N -1] - u [ N ]=0
    # and thus u [ N ]= u [N -1]
    i = len(u) - 1
    d2u_dx2[i] = (u[i] - 2 * u[i] + u[i - 1]) / h ** 2
    return D * d2u_dx2


def advance_time(u, g, dt):
    """ Given the array u , the rate of change array g ,
    and a timestep dt , compute the solution for u
    after t , using simple Euler method . """
    u = u + dt * g
    return u


# show example , quick and dirtly , lots of global variables
dt = 0.01  # step size or time
stepsbeforeupdatinggraph = 20  # plotting is slow
D = 1.  # Diffusion coefficient
stepsdone = 0  # keep track of iterations


def do_steps(j, nsteps=stepsbeforeupdatinggraph):
    """ Function called by F un cA ni ma ti on class . Computes
    nsteps iterations , i . e . carries forward solution from
    u (x , t_i ) to u (x , t_ { i + nsteps }).
    """
    global u, stepsdone
    for i in range(nsteps):
        g = compute_g(u, D, h)
        u = advance_time(u, g, dt)
        stepsdone += 1
        time_passed = stepsdone * dt
    print(" stepsdone =%5 d , time =%8 gs , total ( u )=%8 g" % (stepsdone, time_passed, total(u)))
    l.set_ydata(u)  # update data in plot
    fig1.canvas.draw()  # redraw the canvas
    return l


fig1 = plt.figure()  # setup animation
l, = plt.plot(x, u, )  # plot initial u (x , t )
# then compute solution and animate
line_ani = animation.FuncAnimation(fig1,
                                   do_steps, range(10000))
plt.show()
