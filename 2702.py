import math

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()


ax_1 = fig.add_subplot(4,3,1)
ax_2 = fig.add_subplot(4,3,3)
ax_3 = fig.add_subplot(4,3,12)
ax_4 = fig.add_subplot(4,3,10)
x = [x/10 for x in range(1,11)]

y_1 = [math.sin(y) for y in x]
ax_1.set_title('ax_1')
ax_1.plot(x,y_1, "red")
ax_1.scatter(x,y_1)
ax_1.scatter(x, y_1)
ax_1.set_xlabel('ось абцис (XAxis)')
ax_1.set_ylabel('ось ординат (sin(x))')
y_2 = [- math.sin(y) for y in x]

ax_2.set_title('ax_2')
ax_2.plot(x,y_2, "black")
ax_2.scatter(x,y_2)
ax_2.scatter(x, y_2)
ax_2.set_xlabel('ось абцис (XAxis)')
ax_2.set_ylabel('ось ординат (-sin(x))')
y_3 = [- math.cos(y) for y in x]

ax_3.set_title('ax_3')
ax_3.plot(x,y_3, "green")
ax_3.scatter(x,y_3)
ax_3.scatter(x, y_3)
ax_3.set_xlabel('ось абцис (XAxis)')
ax_3.set_ylabel('ось ординат (-cos(x))')
y_4 = [math.cos(y) for y in x]

ax_4.set_title('ax_3')
ax_4.plot(x,y_4, "yellow")
ax_4.scatter(x,y_4)
ax_4.scatter(x, y_4)
ax_4.set_xlabel('ось абцис (XAxis)')
ax_4.set_ylabel('ось ординат (cos(x))')

plt.show()