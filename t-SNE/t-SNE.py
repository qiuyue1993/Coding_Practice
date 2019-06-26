import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from tsne import bh_sne
import sys 

with open("data", 'rb') as f:
            if sys.version_info > (3, 0):
                data = pkl.load(f, encoding='latin1')
            else:
                data = pkl.load(f)

data =data.astype('float64')


with open("label", 'rb') as f:
            if sys.version_info > (3, 0):
                y_data = pkl.load(f, encoding='latin1')
            else:
                y_data = pkl.load(f)
classNum = 6
y_data = np.where(y_data==1)[1]*(9.0/classNum)

vis_data = bh_sne(data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

fig = plt.figure()
plt.scatter(vis_x, vis_y, c=y_data, s=1, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
fig.savefig('test.png')
