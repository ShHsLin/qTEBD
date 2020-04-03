import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
sns.set()

if __name__ == '__main__':
    depth = int(sys.argv[1])
    g = 1.4
    h = 0.9045

    for idx in range(1,21):
        try:
            T = idx * 0.5
            plt.semilogy(np.load('data/1d_TFI_g%.4f_h%.4f/L31/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, T, depth)),
                         label='T=%.1f' % T)
            print('plot T %.1f' % T, np.load('data/1d_TFI_g%.4f_h%.4f/L31/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, T, depth))[0])
        except Exception as e:
            print(e)


    plt.legend()
    plt.show()

    # T = 0.2
    # T = float(sys.argv[1])
    # for idx in range(6):
    #     try:
    #         depth = idx
    #         plt.semilogy(np.load('data/1d_TFI_g%.4f_h%.4f/L31/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (T, depth)),
    #                      label='D=%d, T=%.1f' % (depth, T))
    #         print('plot D=%d, T=%.1f' % (depth, T))
    #     except Exception as e:
    #         print(e)


    # plt.legend()
    # plt.show()

