import numpy as np
import matplotlib.pyplot as plt

A = np.load('data_te/1d_TFI_g1.0/L31/circuit_depth2_Niter100_1st_Niter_array.npy')
B = np.load('data_te/1d_TFI_g1.0/L31/circuit_depth3_Niter100_1st_Niter_array.npy')

plt.plot(A)
plt.plot(B)
plt.show()
