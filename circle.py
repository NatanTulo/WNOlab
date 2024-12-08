import numpy as np
import matplotlib.pyplot as plt

t1 = np.arange(0.0, 1.2, 0.02)

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(np.sin(2*np.pi*t1), np.cos(2*np.pi*t1), 'k')
plt.plot(np.sin(2*np.pi*t1)+2, np.cos(2*np.pi*t1), 'k')

plt.grid(True)
plt.show()