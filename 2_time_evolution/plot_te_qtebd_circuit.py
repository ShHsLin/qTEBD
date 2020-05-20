import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append('..')
import misc
import setup
import seaborn as sns
color_set = [sns.color_palette("GnBu_d"),
             sns.color_palette("Blues"),
             sns.cubehelix_palette(8),
             sns.light_palette("green"),
            ]

if __name__ == '__main__':
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    h = float(sys.argv[3])
    order = str(sys.argv[4])
    Hamiltonian = 'TFI'

    plt.close()

    final = True
    if final:
        fig, axes_list = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True)
    else:
        fig, axes_list = plt.subplots(3, 1, figsize=(3.4, 6.8), sharex=True)

    ############################################
    dir_path = 'data_te/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)


    try:
        filename = 'ed_sz_array.npy'
        exact_sz = np.load(dir_path + filename)
        filename = 'ed_dt_array.npy'
        exact_dt = np.load(dir_path + filename)

        ax1 = plt.subplot(311)
        plt.plot(exact_dt[:], exact_sz[:,L//2],'--k', label='exact')
    except:
        if L == 11:
            chi = 32
        else:
            chi = 128

        dir_path = 'data_tebd/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)
        filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)

        filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
        path = dir_path + filename
        t_list = np.load(path)

        filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)
        path = dir_path + filename
        mps_sz = np.load(path)

        axes_list[0].plot(t_list[:500], mps_sz[:500,L//2],'-k', label='exact')

    final_time = []
    for depth in range(1,6):
        final_time.append([])
        for idx, N_iter in enumerate([100000]):
            if N_iter < 100:
                continue

            color = color_set[1][int((depth-1)*1.4)]
            # color = color_set[depth-1][-int(idx*1.5)]

            # if depth == 1:
            #     N_iter = 1
            # else:
            #     N_iter = 100
            # color = color_set[1][depth]


            try:
                dir_path = 'data_te/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)

                try:
                    tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order)
                except:
                    tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order, tmp=False)

                running_idx, my_circuit, E_list, t_list, update_error_list, circuit_sz, circuit_ent, num_iter_array = tuple_loaded

                len_t_list = len(t_list)
                axes_list[0].plot(t_list, circuit_sz[:len_t_list,L//2], '-', color=color, label='$M=%d$' % (depth))
                # plt.plot(t_list, np.abs(circuit_sz[:,L//2] - mps_sz[:len_t_list, L//2]), '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                # plt.semilogy(t_list, np.abs(circuit_sz[:,L//2] - mps_sz[:len_t_list, L//2]), '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                plt.setp(axes_list[0].get_xticklabels(), fontsize=8)

                axes_idx = 1
                if final :
                    pass
                else:
                    axes_list[axes_idx].semilogy(t_list, update_error_list,'.', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                    axes_list[axes_idx].set_ylabel('$1 - \mathcal{E}$')
                    axes_idx += 1

                axes_list[axes_idx].plot(t_list, circuit_ent[:len_t_list, L//2], '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                axes_list[axes_idx].set_ylabel(u'$ S_{ent}^{L/2} $')
                chi = int(2**(depth))
                axes_list[axes_idx].axhline(y=-np.log(1./chi), color='r', linestyle='--', label='bound $\\chi=%d$' % chi)
                axes_list[axes_idx].set_ylim([0, np.amax(circuit_ent[:len_t_list, L//2]) + 0.7])

                final_time[-1].append((N_iter, len(t_list)))
            except Exception as e:
                print(e)

    if not final:
        ax2_1 = axes_list[1].twinx()
        for depth in range(1,6):
            # for idx, N_iter in enumerate([1, 10, 100, 1000]):
            for idx, N_iter in enumerate([100000]):
                if N_iter < 100:
                    continue

                color = color_set[1][int((depth-1)*1.4)]
                # color = color_set[depth-1][-int(idx*1.5)]

                try:
                    dir_path = 'data_te/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)

                    try:
                        tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order)
                    except:
                        tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order, tmp=False)

                    running_idx, my_circuit, E_list, t_list, update_error_list, circuit_sz, circuit_ent, num_iter_array = tuple_loaded

                    # filename = 'circuit_depth%d_Niter%d_%s_dt.npy' % (depth, N_iter, order)
                    # path = dir_path + filename
                    # t_list = np.load(path)

                    # filename = 'circuit_depth%d_Niter%d_%s_error.npy' % (depth, N_iter, order)
                    # path = dir_path + filename
                    # update_error_list = np.load(path)

                    update_error_list = np.array(update_error_list)
                    accum_error = np.abs(1. - np.multiply.accumulate(1 - update_error_list))
                    ax2_1.semilogy(t_list, accum_error,'--', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                    ax2_1.set_ylabel(u'$1 - \prod\mathcal{E}$')
                except Exception as e:
                    print(e)




    # plt.setp(axes_list[0].get_xticklabels(), fontsize=8)
    plt.setp(axes_list[0].get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.1)

    # plt.suptitle('Circuit ' + ', $L=$' + str(L) + ', %s-order' % order)
    axes_list[axes_idx].set_xlabel('$Jt$')

    axes_list[0].set_ylabel('$\\langle \sigma_z^{L/2} \\rangle $')
    axes_list[0].set_zorder(100)
    axes_list[0].legend(fontsize=8, framealpha=1., loc='lower left', bbox_to_anchor=(0., -0.5))
    plt.xlim([0,3.])

    plt.savefig('figure/time_evolv_%s/circuit_L%d_g%.4f_h%.4f_%s.pdf' % (Hamiltonian, L, g, h, order))
    plt.subplots_adjust(left=0.2, top=0.93, bottom=0.15, right=0.93)
    plt.show()

    # plt.figure()
    # for data in final_time:
    #     print(data)
    #     plt.plot(data, 'x-')

    # plt.show()
