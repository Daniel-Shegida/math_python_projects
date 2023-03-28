import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)
xx = 5.5
yy = f(5.5)
print("f(5.5)=", yy)
xnew = np.arange(0, 9, 0.1)
ynew = f(xnew) # use interpolation function returned by
"interp1d"
plt.plot(xnew, ynew, 'g-', label="interpolation")
plt.plot(x, y, 'bo', label="experiment")
# plt.scatter([xx], [yy], s=50, c="r", alpha=1.0,
# ker="D", label="interpolated value")
plt.legend(loc='upper right')
plt.savefig("interp1d_test1-graph.pdf", dpi=300)
plt.show()