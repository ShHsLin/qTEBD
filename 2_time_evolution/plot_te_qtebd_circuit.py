import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append('..')
import misc
import seaborn as sns
color_set = [sns.color_palette("GnBu_d"),
             sns.color_palette("Blues"),
             sns.cubehelix_palette(8),
             sns.light_palette("green"),
            ]

if __name__ == '__main__':
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    order = str(sys.argv[3])
    Hamiltonian = 'TFI'

    plt.close()
    fig=plt.figure(figsize=(8,10))

    ############################################
    dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)


    try:
        filename = 'ed_sz_array.npy'
        exact_sz = np.load(dir_path + filename)
        filename = 'ed_dt_array.npy'
        exact_dt = np.load(dir_path + filename)

        ax1 = plt.subplot(311)
        plt.plot(exact_dt[:], exact_sz[:,L//2],'--k', label='exact')
    except:
        chi = 128
        dir_path = 'data_tebd/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
        filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)

        filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
        path = dir_path + filename
        t_list = np.load(path)

        filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)
        path = dir_path + filename
        mps_sz = np.load(path)

        # ax1 = plt.subplot(311)
        # plt.plot(t_list[:500], mps_sz[:500,L//2],'--k', label='exact')

    final_time = []
    for depth in range(1,5):
        final_time.append([])
        for idx, N_iter in enumerate([1, 10, 100, 1000]):
            if N_iter < 100:
                continue

            color = color_set[depth-1][int(idx*1.5)]
            # if depth == 1:
            #     N_iter = 1
            # else:
            #     N_iter = 100
            # color = color_set[1][depth]


            try:
                dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)

                filename = 'circuit_depth%d_Niter%d_%s_energy.npy' % (depth, N_iter, order)
                path = dir_path + filename
                E_list = np.load(path)

                filename = 'circuit_depth%d_Niter%d_%s_dt.npy' % (depth, N_iter, order)
                path = dir_path + filename
                t_list = np.load(path)

                filename = 'circuit_depth%d_Niter%d_%s_error.npy' % (depth, N_iter, order)
                path = dir_path + filename
                update_error_list = np.load(path)

                filename = 'circuit_depth%d_Niter%d_%s_sz_array.npy' % (depth, N_iter, order)
                path = dir_path + filename
                circuit_sz = np.load(path)

                filename = 'circuit_depth%d_Niter%d_%s_ent_array.npy' % (depth, N_iter, order)
                path = dir_path + filename
                circuit_ent = np.load(path)


                ax1 = plt.subplot(311)
                len_t_list = len(t_list)
                # plt.plot(t_list, circuit_sz[:,L//2], '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                # plt.plot(t_list, np.abs(circuit_sz[:,L//2] - mps_sz[:len_t_list, L//2]), '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                plt.semilogy(t_list, np.abs(circuit_sz[:,L//2] - mps_sz[:len_t_list, L//2]), '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                plt.setp(ax1.get_xticklabels(), fontsize=6)

                ax2 = plt.subplot(312, sharex=ax1)
                plt.semilogy(t_list, update_error_list,'.', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                plt.ylabel('$1 - \mathcal{F}$')

                ax3 = plt.subplot(313, sharex=ax1)
                plt.plot(t_list, circuit_ent[:,L//2], '.', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                plt.ylabel('engtanglement')
                chi = int(2**(depth))
                plt.axhline(y=-np.log(1./chi), color='r', linestyle='--', label='bound $\\chi=%d$' % chi)

                final_time[-1].append((N_iter, len(t_list)))
            except Exception as e:
                print(e)

    ax2_1 = ax2.twinx()
    for depth in range(1,5):
        for idx, N_iter in enumerate([1, 10, 100, 1000]):
            if N_iter < 100:
                continue

            color = color_set[depth-1][int(idx*1.5)]

            try:
                dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)

                filename = 'circuit_depth%d_Niter%d_%s_dt.npy' % (depth, N_iter, order)
                path = dir_path + filename
                t_list = np.load(path)

                filename = 'circuit_depth%d_Niter%d_%s_error.npy' % (depth, N_iter, order)
                path = dir_path + filename
                update_error_list = np.load(path)

                accum_error = np.abs(1 - np.multiply.accumulate(1 - update_error_list))
                ax2_1.semilogy(t_list, accum_error,'-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                ax2_1.set_ylabel(u'$1 - \prod\mathcal{F}$')
            except Exception as e:
                print(e)




    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0)

    plt.suptitle('Circuit ' + ', $L=$' + str(L) + ', %s-order' % order)
    plt.xlabel('$\\tau$')
    plt.subplot(311)
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    plt.ylabel('$< S_z^{L/2} >$')
    plt.legend()
    plt.savefig('figure/time_evolv_%s/circuit_L%d_g%.1f_%s.png' % (Hamiltonian, L, g, order))
    plt.show()

    plt.figure()
    for data in final_time:
        print(data)
        plt.plot(data, 'x-')

    plt.show()
