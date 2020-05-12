import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
# sns.set()
color_set = [sns.color_palette("GnBu_d"),
             sns.color_palette("Blues"),
             sns.cubehelix_palette(8),
             sns.light_palette("green"),
            ]



if __name__ == '__main__':
    # depth = int(sys.argv[1])
    g = 1.0
    h = 0. #0.9045
    chi = 32

    for idx in range(2,10):
        try:
            # T = idx * 0.5
            if np.isclose(g, 1.):
                T = 2.5
            else:
                T = 4.0

            depth = idx
            plt.semilogy(np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, chi, T, depth)),
                         label='original, depth=%d' % (depth), color=color_set[1][idx])
            plt.semilogy(np.load('data_exponential/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, chi, T, depth)),
                         label='exp, depth=%d' % (depth), color=color_set[3][idx])
            plt.semilogy(np.load('data_linear/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, chi, T, depth)),
                         label='linear, depth=%d' % (depth), color=color_set[2][idx])
            print('plot T %.1f' % T, np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, chi, T, depth))[0])
        except Exception as e:
            print(e)


    plt.title('g=%.4f, h=%.4f' % (g, h))
    plt.xlabel('Number of iterations')
    plt.ylabel('$1-\mathcal{F}$')
    plt.legend()
    plt.savefig('figure/compare_init_g%.4f_h%.4f.png' % (g,h))
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

