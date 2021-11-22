import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scat = ax.scatter([1,2,3],[1,2,3],[1,2,3], marker='o', c=[1,2,3])

ax.set_xlim((-1,1))
ax.set_ylim((-1,1))
ax.set_zlim((-1,1))

def update(i):
    x,y,z = np.array([[1,1,1], [0,0,0], [1,2,3.]]).transpose()*np.sin(i/10.)
    print(z)
    scat.set_offsets(([1.,0.],[1.,0.], [0.5, 0.5]))
    scat.set_3d_properties(z, 'z')

anim = animation.FuncAnimation(fig, update)
plt.show()
