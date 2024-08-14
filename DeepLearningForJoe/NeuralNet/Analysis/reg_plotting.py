import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.optimize as optimize
import mat73
matplotlib.use('Qt5Agg')

data = mat73.loadmat(r'C:\Joe Evans\University\Computing\Summer Project\DeepLearningForJoe\SimulateData\Experiments\regularisation_picker\regularisation_information.mat')

fl = sum(data['all_datafl'])
x = sum(data['all_datax'])
reg = np.log10(data['regularisations'])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(fl, x, reg)
ax.set_xlabel('fl')
ax.set_ylabel('x')
ax.set_zlabel('reg')

plt.show()
