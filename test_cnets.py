import cnets 
import numpy as np
import matplotlib.pyplot as plt

links = [[0,1, 1.],
         [1,2, 1.],
         [2,3, 1.], 
         [3,0, 1.],

         [3,1, 1.4142],
         [2,0, 1.4142]]

values = [8.7, 9.8, 11.9, 5.]
cnets.init_network(links, values, 2)
cnets.MDE(1., 1000000)
net = np.array(cnets.get_positions()).transpose()
print(cnets.get_distanceSM())
plt.plot(*net, ls='', marker='o')
plt.show()
