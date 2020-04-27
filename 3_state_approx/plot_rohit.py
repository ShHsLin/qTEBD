import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import sys
sys.path.append('..')
import qTEBD

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

    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]
    product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]


    exact_sz = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_sz_array.npy' % (g, h, chi))
    exact_ent = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_ent_array.npy' % (g, h, chi))
    exact_E = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_energy.npy' % (g, h, chi))
    exact_t = np.load('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/mps_chi%d_1st_dt.npy' % (g, h, chi))

    if np.isclose(g, 1.0):
        rohit_wf_dir =  '/space/ge38huj/state_approximation/mm_initial_guesses_chi32/'
        T = 2.5
    else:
        rohit_wf_dir = '/space/ge38huj/state_approximation/mm_initial_guesses_long_chi32/'
        T = 4.0


    exact_state = pickle.load(open('../2_time_evolution/data_tebd/1d_TFI_g%.4f_h%.4f/L31/wf_chi%d_1st/T%.1f.pkl' % (g, h, chi, T), 'rb'))

    depth_list = []
    color = color_set[2][0]
    fidelity_error_list = []
    diff_sz_list = []
    ent_list = []
    for idx, depth in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        try:
            exact_idx = int(T * 100)

            rohit_circuit = pickle.load(open(rohit_wf_dir + '%d_layers_exponential_schedule_T%.1f.pkl' % (depth, T),'rb'))
            # rohit_circuit = pickle.load(open(rohit_wf_dir + '%d_layers_linear_schedule_T%.1f.pkl' % (depth, T),'rb'))

            mps_of_layer = qTEBD.circuit_2_mps(rohit_circuit, product_state)
            mps_of_last_layer = [A.copy() for A in mps_of_layer[depth]]


            fidelity_reached = np.abs(qTEBD.overlap(exact_state, mps_of_last_layer))**2
            f_error = 1. - fidelity_reached
            print('f_error : ', f_error)
            fidelity_error_list.append(f_error)

            sz_data = qTEBD.expectation_values_1_site(mps_of_last_layer, Sz_list)
            abs_diff_sz = np.abs(sz_data[L//2] - exact_sz[exact_idx, L//2])
            diff_sz_list.append(abs_diff_sz)

            ent_data = qTEBD.get_entanglement(mps_of_last_layer)[L//2]
            ent_list.append(ent_data)
            depth_list.append(depth)

        except Exception as e:
            print(e)


    ax1 = plt.subplot(3,1,1)
    # plt.semilogy(depth_list, fidelity_error_list, 'x--', color=color, label='R depth=%d' % depth)
    plt.plot(depth_list, fidelity_error_list, 'x--', color=color, label='depth=%d' % depth)
    plt.yscale('log')
    ax2 = plt.subplot(3,1,2, sharex=ax1)
    # plt.semilogy(depth_list, diff_sz_list, 'x--', color=color, label='R depth=%d' % depth)
    plt.plot(depth_list, diff_sz_list, 'x--', color=color, label='R depth=%d' % depth)
    plt.yscale('log')
    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plt.plot(depth_list, ent_list, 'x--', color=color, label='R depth=%d' % depth)




    depth_list = []
    color = color_set[2][2]
    fidelity_error_list = []
    diff_sz_list = []
    ent_list = []
    for idx, depth in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        try:
            exact_idx = int(T * 100)

            f_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi32/T%.1f/circuit_depth%d_Niter100000_1st_error.npy' % (g, h, T, depth))
            fidelity_error_list.append(f_data[-1])

            sz_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi32/T%.1f/circuit_depth%d_Niter100000_1st_sz_array.npy' % (g, h, T, depth))[-1]
            abs_diff_sz = np.abs(sz_data[L//2] - exact_sz[exact_idx, L//2])
            diff_sz_list.append(abs_diff_sz)

            ent_data = np.load('data/1d_TFI_g%.4f_h%.4f/L31_chi32/T%.1f/circuit_depth%d_Niter100000_1st_ent_array.npy' % (g, h, T, depth))[-1, L//2]
            ent_list.append(ent_data)
            depth_list.append(depth)

        except Exception as e:
            print(e)


    ax1 = plt.subplot(3,1,1)
    plt.semilogy(depth_list, fidelity_error_list, 'x--', color=color, label='depth=%d' % depth)
    ax2 = plt.subplot(3,1,2, sharex=ax1)
    plt.semilogy(depth_list, diff_sz_list, 'x--', color=color, label='depth=%d' % depth)
    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plt.plot(depth_list, ent_list, 'x--', color=color, label='depth=%d' % depth)

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
    ent = exact_ent[np.argwhere(np.isclose(exact_t, T))[0][0], L//2]
    # plt.plot(exact_t[:500], exact_ent[:500,L//2], '--', label='exact')
    plt.axhline(y=ent, color='r', linestyle='--', label='exact')


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
    plt.xlabel(u'depth')
    plt.legend()
    ax1 = plt.subplot(3,1,1)
    plt.title(u"$g = %.4f, h = %.4f$" % (g, h))
    # plt.savefig('figure/g%.4f_h%.4f.png' % (g, h))
    plt.show()


