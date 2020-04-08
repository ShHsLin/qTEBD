import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import seaborn as sns
# sns.set()

if __name__ == '__main__':
    L = 31
    g = 1.0
    h = 0. # 9045
    chi = 128
    order = '1st'

    exact_sz = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_sz_array.npy' % (g, h, chi))
    exact_ent = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_ent_array.npy' % (g, h, chi))
    exact_E = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_energy.npy' % (g, h, chi))
    exact_t = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_dt.npy' % (g, h, chi))

    for depth in [2, 3, 4, 5]:
        try:
            chi = 2 ** depth
            fidelity_error_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31/approx_mps/mps_chi%d_%s_error.npy' % (g, h, chi, order))

            ent_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31/approx_mps/mps_chi%d_%s_ent_array.npy' % (g, h, chi, order))[-1, L//2]
            t_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31/approx_mps/mps_chi%d_%s_dt.npy' % (g, h, chi, order))

        except Exception as e:
            print(e)


        plt.semilogy(t_list, fidelity_error_list, '--', label='chi=%d' % chi)






    plt.ylabel(u'$1-\mathcal{F}$')

    plt.xlabel(u'T')
    plt.legend()
    plt.title("Truncating from MPS with $\chi=128$")
    plt.savefig('mps_g%.4f_h%.4f.png' % (g, h))
    plt.show()


