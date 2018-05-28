
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


lenx = 1
lent = 1
deltat = 1 / 100
deltax = 1 / 50
Nx = 100
Nt = int(lent / deltat)
st = deltat / deltax

u = np.zeros([Nx, Nt])

u[0, :] = 0
u[-1, :] = 0

for i in range(1, Nx - 1):
    u[i, 0] = np.sin(math.pi) * i


x = list(range(Nx))
y = list(range(Nt))

X, Y = np.meshgrid(x, y)


def functz(u):
    z = u[X, Y]
    return z


for i in range(1, Nx - 1):
    for j in range(1, Nt - 1):
        u[i, j + 1] = st * u[i + 1, j] + (1 - st) * u[i, j]

'''
print(u.shape)
plt.plot(u[1], u[2])
plt.show()

'''

Z = functz(u)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x, y, Z, color='r')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U')

plt.show()
