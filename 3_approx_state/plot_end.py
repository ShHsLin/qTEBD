import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import seaborn as sns
# sns.set()

if __name__ == '__main__':
    L = 31
    exact_sz = np.load('../2_time_evolution/data_tebd/1d_TFI_g1.0/L31/mps_chi128_1st_sz_array.npy')
    exact_ent = np.load('../2_time_evolution/data_tebd/1d_TFI_g1.0/L31/mps_chi128_1st_ent_array.npy')
    exact_E = np.load('../2_time_evolution/data_tebd/1d_TFI_g1.0/L31/mps_chi128_1st_energy.npy')
    exact_t = np.load('../2_time_evolution/data_tebd/1d_TFI_g1.0/L31/mps_chi128_1st_dt.npy')

    for depth in [2, 3, 4, 5]:
        fidelity_error_list = []
        diff_sz_list = []
        ent_list = []
        t_list = []

        for idx in range(0,50):
            try:
                T = idx * 0.1
                exact_idx = int(idx * 10)

                f_data = np.load('data/1d_TFI_g1.0/L31/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (T, depth))
                fidelity_error_list.append(f_data[-1])

                sz_data = np.load('data/1d_TFI_g1.0/L31/T%.1f/circuit_depth%d_Niter100000_1st_sz_array.npy' % (T, depth))[-1]
                abs_diff_sz = np.abs(sz_data[L//2] - exact_sz[exact_idx, L//2])
                diff_sz_list.append(abs_diff_sz)

                ent_data = np.load('data/1d_TFI_g1.0/L31/T%.1f/circuit_depth%d_Niter100000_1st_ent_array.npy' % (T, depth))[-1, L//2]
                ent_list.append(ent_data)
                t_list.append(T)

            except Exception as e:
                print(e)


        ax1 = plt.subplot(3,1,1)
        plt.semilogy(t_list, fidelity_error_list, 'x--', label='depth=%d' % depth)
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        plt.semilogy(t_list, diff_sz_list, 'x--', label='depth=%d' % depth)
        ax3 = plt.subplot(3,1,3, sharex=ax1)
        plt.plot(t_list, ent_list, 'x--', label='depth=%d' % depth)


    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plt.plot(exact_t[:500], exact_ent[:500,L//2], '--', label='exact')

    scale_x = 10
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    # ax1.xaxis.set_major_formatter(ticks_x)
    # ax2.xaxis.set_major_formatter(ticks_x)
    # ax3.xaxis.set_major_formatter(ticks_x)

    ax1 = plt.subplot(3,1,1)
    plt.ylabel(u'$1-\mathcal{F}$')
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0)
    plt.ylabel(u'$|\Delta Sz|$')

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plt.ylabel(u'entanglement')
    plt.xlabel(u'T')
    plt.legend()
    plt.show()


