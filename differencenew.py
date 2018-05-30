import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


lenx = 1
lent = 1
deltat = 1 / 100
deltax = 1 / 100
Nt = 100
Nx = int(lenx / deltax)
st = deltat / deltax

u = np.zeros([Nx, Nt])

u[0, :] = 0
u[-1, :] = 0


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


ubot = []
for i in drange(0.00, 1.00, deltax):
    ubot.append(np.sin(math.pi * i))

for k in range(Nx):
    u[k, 0] = ubot[k]
    print(u[k, 0])


for j in range(Nt - 1):
    for i in range(Nx - 2):
        u[i, j + 1] = st * (u[i + 1, j] - u[i - 1, j]) + u[i, j]


x = list(range(Nx))
y = list(range(Nt))

X, Y = np.meshgrid(x, y)


def functz(u):
    z = u[X, Y]
    return z


Z = functz(u)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, color='r')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U')

plt.show()
