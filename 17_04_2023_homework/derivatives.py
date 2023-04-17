import math

EPS = 1  # very large EPS to provoke inaccuracy


def forwarddiff(f, x, h=EPS):
    # df / dx = ( f ( x + h ) - f ( x ) )/ h + O ( h )
    return (f(x + h) - f(x)) / h


def backwarddiff(f, x, h=EPS):
    # df / dx = ( f ( x ) - f (x - h ) )/ h + O ( h )
    return (f(x) - f(x - h)) / h


def centraldiff(f, x, h=EPS):
    # df / dx = ( f ( x + h ) - f (x - h ))/ h + O ( h ^2)
    return (f(x + h) - f(x - h)) / (2 * h)


if True:
    # create example plot
    import pylab
    import numpy as np

    a = 0  # left and
    b = 5  # right limits for x
    N = 11  # steps


    def f(x):
        """ Our test funtion with
        convenient property that
        df / dx = f """
        return np.exp(x)


    xs = np.linspace(a, b, N)
    forward = []
    forward_small_h = []
    central = []
    for x in xs:
        forward.append(forwarddiff(f, x))
        central.append(centraldiff(f, x))
        forward_small_h.append(forwarddiff(f, x, h=1 * math.e - 4))

    pylab.figure(figsize=(6, 4))
    pylab.axis([a, b, 0, np.exp(b)])
    pylab.plot(xs, forward, '^', label='forward h = % g ' % EPS)
    pylab.plot(xs, central, 'x', label=' central h = % g ' % EPS)
    pylab.plot(xs, forward_small_h, 'o',
               label='forward h = % g' % int(1 * math.e - 4))
    xsfine = np.linspace(a, b, N * 100)

    pylab.plot(xsfine, f(xsfine), '-', label='exact')
    pylab.grid()
    pylab.legend(loc='upper left')
    pylab.xlabel(" x")
    pylab.ylabel(" df / dx (x)")
    pylab.title(" Approximations of df / dx for f(x )= exp ( x)")
    pylab.plot()
    pylab.savefig('central - and - forward - difference.pdf')
    pylab.show()
