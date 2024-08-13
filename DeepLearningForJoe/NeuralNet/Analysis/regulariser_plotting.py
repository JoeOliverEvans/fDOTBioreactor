import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.optimize as optimize

matplotlib.use('Qt5Agg')


def linear(data, a1, a2, a3):
    [data1, data2] = data
    return a1 * data1 + a2 * data2 + a3


fl = np.array([0.0099,
               0.0082,
               0.0054,
               0.0031,
               0.0114,
               0.0144,
               0.0058,
               0.0094,
               0.0039,
               0.0062,
               0.0025
               ])

x = np.array([0.5609,
              0.5674,
              0.5824,
              0.5763,
              0.5607,
              0.5587,
              0.5734,
              0.5577,
              0.5771,
              0.5662,
              0.5895
              ])

reg = np.array([1,
                0.05,
                0.005,
                0.001,
                0.01,
                0.05,
                0.001,
                0.1,
                0.01,
                0.1,
                0.005
                ])

reg = np.log10(reg)
input = [fl, x]
[popt, pcov] = optimize.curve_fit(linear, input, reg)
print(popt)
print(pcov)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X = np.linspace(0.004, 0.014, 10)
Y = np.linspace(0.560, 0.580, 10)
X, Y = np.meshgrid(X, Y)


ax.scatter(fl, x, reg)
ax.plot_surface(X, Y, linear([X, Y], *popt), color='red')
ax.set_xlabel('fl')
ax.set_ylabel('x')
ax.set_zlabel('reg')

plt.show()
