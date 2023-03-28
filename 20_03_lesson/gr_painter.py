import math

from matplotlib import pyplot as plt


def x_y_res_painter(t, tMax, vy_vac, vy_vac_min, vy_vac_max,  vy_res, vy_res_min, vy_res_max, vx_vac, vx_vac_min, vx_vac_max, vx_res, vx_res_min,  vx_res_max,  y_vac, y_res, x_vac, x_res, ):
    fig = plt.figure()

    ax_1 = fig.add_subplot(4, 3, 1)
    ax_2 = fig.add_subplot(4, 3, 3)
    ax_3 = fig.add_subplot(4, 3, 12)
    ax_4 = fig.add_subplot(4, 3, 10)


    print(t)
    print('t')
    print(vy_vac)
    print('vy')
    ax_1.set_title('Vz')
    ax_1.plot(t, vy_vac, "red")
    ax_1.plot(t, [0.0] * vy_vac, 'g-', linewidth=1)
    ax_1.axis([0, tMax, vy_vac_min, vy_vac_max])

    ax_2.set_title('Vzres')
    ax_2.plot(t, vy_vac, "red")
    ax_2.plot(t, [0.0] * vy_res, 'g-', linewidth=1)
    ax_2.axis([0, tMax, vy_res_min, vy_res_max])

    ax_3.set_title('Vx')
    ax_3.plot(t, vx_vac, "red")
    ax_3.plot(t, [0.0] * vy_vac, 'g-', linewidth=1)
    # ax_3.axis([0, tMax, 0, vx_vac_max])

    ax_4.set_title('Vxres')
    ax_4.plot(t, vx_res, "red")
    ax_4.plot(t, [0.0] * vy_vac, 'g-', linewidth=1)
    ax_4.axis([0, tMax, 0, vx_res_max])

    plt.show()