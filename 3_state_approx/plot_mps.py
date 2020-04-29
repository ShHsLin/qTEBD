import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import seaborn as sns
# sns.set()

if __name__ == '__main__':
    L = 31
    g = float(sys.argv[1])
    h = float(sys.argv[2])
    chi = int(sys.argv[3])
    order = '1st'

    exact_sz = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_sz_array.npy' % (g, h, chi))
    exact_ent = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_ent_array.npy' % (g, h, chi))
    exact_E = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_energy.npy' % (g, h, chi))
    exact_t = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_dt.npy' % (g, h, chi))

    for depth in [2, 3, 4, 5]:
        try:
            new_chi = 2 ** depth
            fidelity_error_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/approx_mps/mps_chi%d_%s_error.npy' % (g, h, chi, new_chi, order))

            ent_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/approx_mps/mps_chi%d_%s_ent_array.npy' % (g, h, chi, new_chi, order))[-1, L//2]
            t_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/approx_mps/mps_chi%d_%s_dt.npy' % (g, h, chi, new_chi, order))
            plt.semilogy(t_list, fidelity_error_list, '--', label=u'$\chi_{trunc}=%d$' % new_chi)

        except Exception as e:
            print(e)








    plt.ylabel(u'$1-\mathcal{F}$')

    plt.xlabel(u'T')
    plt.legend()
    plt.title("Truncating from MPS with $\chi=%d$" % chi)
    # plt.savefig('figure/mps_g%.4f_h%.4f.png' % (g, h))
    plt.show()


