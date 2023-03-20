import math

from matplotlib import pyplot as plt


def x_y_res_painter(t, vy_vac, vy_res, vx_vac, vx_res, y_vac, y_res, x_vac, x_res, ):
    fig = plt.figure()

    ax_1 = fig.add_subplot(4, 3, 1)
    ax_2 = fig.add_subplot(4, 3, 3)
    ax_3 = fig.add_subplot(4, 3, 12)
    ax_4 = fig.add_subplot(4, 3, 10)
    x = t

    # y_1 = [math.sin(y) for y in x]
    ax_1.set_title('ax_1')
    ax_1.plot(x, vy_vac, "red")
    # ax_1.scatter(x,vy_vac)
    # ax_1.scatter(x, vy_vac)
    ax_1.set_xlabel('ось абцис (XAxis)')
    ax_1.set_ylabel('ось ординат (sin(x))')
    y_2 = [- math.sin(y) for y in x]

    ax_2.set_title('ax_2')
    ax_2.plot(x, vy_res, "black")
    # ax_2.scatter(x, vy_res)
    # ax_2.scatter(x, vy_res)
    ax_2.set_xlabel('ось абцис (XAxis)')
    ax_2.set_ylabel('ось ординат (-sin(x))')
    y_3 = [- math.cos(y) for y in x]

    ax_3.set_title('ax_3')
    ax_3.plot(x, vx_vac, "green")
    # ax_3.scatter(x, vx_vac)
    # ax_3.scatter(x, vx_vac)
    ax_3.set_xlabel('ось абцис (XAxis)')
    ax_3.set_ylabel('ось ординат (-cos(x))')
    y_4 = [math.cos(y) for y in x]

    ax_4.set_title('ax_3')
    ax_4.plot(x, vx_res, "yellow")
    # ax_4.scatter(x, vx_res)
    # ax_4.scatter(x, vx_res)
    ax_4.set_xlabel('ось абцис (XAxis)')
    ax_4.set_ylabel('ось ординат (cos(x))')

    plt.show()