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
    if np.isclose(g, 1.0):
        chi = 128
    else:
        chi = 32

    order = '1st'
    plot_topic = 'f'
    num_frame = 2
    if num_frame == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, sharex='row');    ax2 = ax1

    exact_sz = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_sz_array.npy' % (g, h, chi))
    exact_ent = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_ent_array.npy' % (g, h, chi))
    exact_E = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_energy.npy' % (g, h, chi))
    exact_t = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_dt.npy' % (g, h, chi))

    cutoff_time = 4.
    dt = 0.1
    T_idx_end = int(cutoff_time / dt) + 1
    for idx, depth in enumerate([2, 3, 4, 5]):
        color = color_set[2][int(idx*1.5)]
        fidelity_error_list = []
        diff_sz_list = []
        ent_list = []
        num_iter_list = []
        t_list = []

        for idx in range(0, T_idx_end):
            try:
                T = idx * dt
                exact_idx = int(idx * 10)

                f_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, chi, T, depth))
                fidelity_error_list.append(f_data[-1])
                num_iter_list.append(len(f_data))

                sz_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_sz_array.npy' % (g, h, chi, T, depth))[-1]
                abs_diff_sz = np.abs(sz_data[L//2] - exact_sz[exact_idx, L//2])
                diff_sz_list.append(abs_diff_sz)

                ent_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/T%.1f/circuit_depth%d_Niter100000_1st_ent_array.npy' % (g, h, chi, T, depth))[-1, L//2]
                ent_list.append(ent_data)
                t_list.append(T)

            except Exception as e:
                print(e)


        if plot_topic == 'f':
            # ax1 = plt.subplot(2,1,1)
            ax1.semilogy(t_list, fidelity_error_list, 'x--', color=color, label='depth=%d' % depth)

            # ax2 = plt.subplot(2,1,2, sharex=ax1)
            # plt.semilogy(t_list, diff_sz_list, 'x--', color=color, label='depth=%d' % depth)
            # plt.semilogy(t_list, num_iter_list, 'x--', color=color, label='depth=%d' % depth)
        elif plot_topic == 'ent':
            # ax1 = plt.subplot(2,1,1)
            ax1.plot(t_list, ent_list, 'x--', color=color, label='depth=%d' % depth)

    for idx, depth in enumerate([2, 3, 4, 5, ]):
        chi = 128
        color = color_set[2][int(idx*1.5)]
        try:
            new_chi = 2 ** depth
            fidelity_error_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/approx_mps/mps_chi%d_%s_error.npy' % (g, h, chi, new_chi, order))
            # sz_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31/circuit_depth%d_Niter100000_1st_sz_array.npy' % (g, h, depth))[-1]
            # abs_diff_sz = np.abs(sz_data[L//2] - exact_sz[exact_idx, L//2])
            # diff_sz_list.append(abs_diff_sz)

            ent_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/approx_mps/mps_chi%d_%s_ent_array.npy' % (g, h, chi, new_chi, order))[:, L//2]
            t_list = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi%d/approx_mps/mps_chi%d_%s_dt.npy' % (g, h, chi, new_chi, order))

            fidelity_error_list = fidelity_error_list[t_list <= cutoff_time]
            ent_list = ent_list[t_list <= cutoff_time]
            t_list = t_list[t_list <= cutoff_time]

            if plot_topic == 'f':
                # ax1 = plt.subplot(2,1,1)
                # ax2 = plt.subplot(2,1,2, sharex=ax1)
                ax2.semilogy(t_list, fidelity_error_list, '-', color=color, label=u'$\chi_{trunc}=%d$' % new_chi)
                # ax2 = plt.subplot(2,1,2, sharex=ax1)
                # plt.semilogy(t_list, diff_sz_list, '-', label='depth=%d' % depth)
            elif plot_topic == 'ent':
                # ax2 = plt.subplot(2,1,2, sharex=ax1)
                ax2.plot(t_list, ent_list, '-', label=u'$\chi_{trunc}=%d$' % new_chi)

        except Exception as e:
            print(e)


    if plot_topic == 'ent':
        # ax1 = plt.subplot(2,1,1)
        ax1.plot(exact_t[:500], exact_ent[:500,L//2], 'k--', label='exact')
        if num_frame == 2:
            # ax2 = plt.subplot(2,1,1, sharex=ax1)
            ax2.plot(exact_t[:500], exact_ent[:500,L//2], 'k--', label='exact')


    # ax1 = plt.subplot(2,1,1)
    ax1.legend(fontsize=6)


    scale_x = 10
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    # ax1.xaxis.set_major_formatter(ticks_x)
    # ax2.xaxis.set_major_formatter(ticks_x)
    # ax3.xaxis.set_major_formatter(ticks_x)

    if plot_topic == 'f':
        # ax1 = plt.subplot(2,1,1)
        ax1.set_title(u"$g = %.4f, h = %.4f$" % (g, h))
        ax1.set_ylabel(u'$1-\mathcal{F}$')
        plt.setp(ax1.get_xticklabels(), visible=False)
        # plt.setp(ax1.get_xticklabels(), fontsize=6)
        ax1.set_ylim([1e-8, 1e-0])
        ax1.axhline(1e-4,color='r',linestyle='--')

        # ax2 = plt.subplot(2,1,2, sharex=ax1)
        plt.subplots_adjust(hspace=0)
        ax2.set_ylim([1e-8, 1e-0])
        plt.axhline(1e-4,color='r',linestyle='--')
        ax2.set_ylabel(u'$1-\mathcal{F}$')
        ax2.set_xlabel(u'T')
        ax2.legend(fontsize=6)
        plt.savefig('figure/cf%d_mps_circuit_%s_g%.4f_h%.4f.svg' % (num_frame, plot_topic, g, h))
        plt.savefig('figure/cf%d_mps_circuit_%s_g%.4f_h%.4f.pdf' % (num_frame, plot_topic, g, h))
        plt.show()
    elif plot_topic == 'ent':
        # ax1 = plt.subplot(2,1,1)
        ax1.set_title(u"$g = %.4f, h = %.4f$" % (g, h))
        ax1.set_ylabel(u'entanglement')
        if num_frame == 2:
            plt.setp(ax1.get_xticklabels(), visible=False)

        #plt.ylim([])

        # ax2 = plt.subplot(2,1,2, sharex=ax1)
        plt.subplots_adjust(hspace=0)
        #plt.ylim()
        ax2.set_ylabel(u'entanglement')
        ax2.set_xlabel(u'T')
        ax2.legend(fontsize=6)

        plt.savefig('figure/cf%d_mps_circuit_%s_g%.4f_h%.4f.svg' % (num_frame, plot_topic, g, h))
        plt.savefig('figure/cf%d_mps_circuit_%s_g%.4f_h%.4f.pdf' % (num_frame, plot_topic, g, h))
        plt.show()


