import matplotlib.pyplot as plt
import edward as ed
import numpy as np


y = np.random.randn(20)
y_rep = np.random.randn(20, 20)

ed.ppc_density_plot(y, y_rep)
plt.show()
