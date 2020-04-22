import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    L = 31
    g = float(sys.argv[1]) # 1.4
    h = float(sys.argv[2]) # 0.9045
    chi = 32
    order = '1st'

    exact_sz = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_sz_array.npy' % (g, h, chi))
    exact_ent = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_ent_array.npy' % (g, h, chi))
    exact_E = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_energy.npy' % (g, h, chi))
    exact_t = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_dt.npy' % (g, h, chi))

    for idx, depth in enumerate([2, 3, 4, 5, 6,]):
        color = color_set[2][idx]
        fidelity_error_list = []
        diff_sz_list = []
        ent_list = []
        t_list = []

        for idx in range(0, 41):
            try:
                T = idx * 0.1
                exact_idx = int(idx * 10)

                f_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, chi, T, depth))
                fidelity_error_list.append(f_data[-1])

                sz_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_sz_array.npy' % (g, h, chi, T, depth))[-1]
                abs_diff_sz = np.abs(sz_data[L//2] - exact_sz[exact_idx, L//2])
                diff_sz_list.append(abs_diff_sz)

                ent_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_ent_array.npy' % (g, h, chi, T, depth))[-1, L//2]
                ent_list.append(ent_data)
                t_list.append(T)

            except Exception as e:
                print(e)


        ax1 = plt.subplot(3,1,1)
        plt.semilogy(t_list, fidelity_error_list, 'x--', color=color, label='depth=%d' % depth)
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        plt.semilogy(t_list, diff_sz_list, 'x--', color=color, label='depth=%d' % depth)
        ax3 = plt.subplot(3,1,3, sharex=ax1)
        plt.plot(t_list, ent_list, 'x--', color=color, label='depth=%d' % depth)

    # for idx, depth in enumerate([2, 3, 4, 5, 6, 7, 8]):
    #     color = color_set[2][idx]
    #     try:
    #         chi = 2 ** depth
    #         fidelity_error_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31/approx_mps/mps_chi%d_%s_error.npy' % (g, h, chi, order))
    #         # sz_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31/circuit_depth%d_Niter100000_1st_sz_array.npy' % (g, h, depth))[-1]
    #         # abs_diff_sz = np.abs(sz_data[L//2] - exact_sz[exact_idx, L//2])
    #         # diff_sz_list.append(abs_diff_sz)

    #         ent_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31/approx_mps/mps_chi%d_%s_ent_array.npy' % (g, h, chi, order))[-1, L//2]
    #         t_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31/approx_mps/mps_chi%d_%s_dt.npy' % (g, h, chi, order))

    #     except Exception as e:
    #         print(e)


    #     ax1 = plt.subplot(3,1,1)
    #     plt.semilogy(t_list, fidelity_error_list, '-', color=color, label='chi=%d' % chi)
    #     # ax2 = plt.subplot(3,1,2, sharex=ax1)
    #     # plt.semilogy(t_list, diff_sz_list, 'x--', label='depth=%d' % depth)
    #     # ax3 = plt.subplot(3,1,3, sharex=ax1)
    #     # plt.plot(t_list, ent_list, 'x--', label='depth=%d' % depth)



    # ax1 = plt.subplot(3,1,1)
    # plt.legend()




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
    ax1 = plt.subplot(3,1,1)
    plt.title(u"$g = %.4f, h = %.4f$" % (g, h))
    plt.savefig('g%.4f_h%.4f.png' % (g, h))
    plt.show()


