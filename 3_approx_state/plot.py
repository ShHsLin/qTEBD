import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
sns.set()

if __name__ == '__main__':
    depth = int(sys.argv[1])

    for idx in range(20,30):
        try:
            T = idx * 0.1
            plt.semilogy(np.load('data/1d_TFI_g1.0/L31/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (T, depth)),
                         label='T=%.1f' % T)
            print('plot T %.1f' % T, np.load('data/1d_TFI_g1.0/L31/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (T, depth))[0])
        except Exception as e:
            print(e)


    plt.legend()
    plt.show()

    # T = 0.2
    # T = float(sys.argv[1])
    # for idx in range(6):
    #     try:
    #         depth = idx
    #         plt.semilogy(np.load('data/1d_TFI_g1.0/L31/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (T, depth)),
    #                      label='D=%d, T=%.1f' % (depth, T))
    #         print('plot D=%d, T=%.1f' % (depth, T))
    #     except Exception as e:
    #         print(e)


    # plt.legend()
    # plt.show()

