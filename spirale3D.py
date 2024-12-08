from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# create 3d axes
fig = plt.figure()
ax = plt.axes(projection='3d')

# cordiates for spiral
# z = np.linspace(0, 15, 1000)
# x = np.sin(z)
# y = np.cos(z)

y=np.linspace(0, 75, 1000)
x=np.sin(y)
z=np.cos(y)
ax.plot3D(x, y, z, 'red')

plt.show()