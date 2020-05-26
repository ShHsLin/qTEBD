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

markersize = 4.
linewidth = 1


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
        fig2, axes2  = plt.subplots(1, 1, figsize=(3.4, 1.7))
    else:
        fig, axes_list = plt.subplots(3, 1, figsize=(3.4, 4.5), sharex=True)

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

        filename = 'mps_chi%d_%s_ent_array.npy' % (chi, order)
        path = dir_path + filename
        mps_ent = np.load(path)

        axes_list[0].plot(t_list[:500], mps_sz[:500,L//2], '-k', linewidth=linewidth, label='exact')
        if final:
            idx = 1
        else:
            idx = 2

        axes_list[idx].plot(t_list[:500], mps_ent[:500,L//2], '-k', linewidth=linewidth, label='exact')


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

                circuit_sz = circuit_sz.real
                len_t_list = len(t_list)

                ##### Compute accum_error #####
                update_error_list = np.array(update_error_list)
                accum_error = np.abs(1. - np.multiply.accumulate(1 - update_error_list))
                ##### Determine to where one plot the plot #####
                err_idx = np.argmax(accum_error > 1e-2)
                if err_idx == 0:
                    err_idx = len(accum_error)  ## This is a artificial fix, when no condition is satisfied, and 0 returned. 

                end_idx = np.amin([err_idx, len_t_list])

                axes_list[0].plot(t_list[:end_idx], circuit_sz[:end_idx,L//2], '-', color=color, label='$M=%d$' % (depth))
                # plt.plot(t_list, np.abs(circuit_sz[:,L//2] - mps_sz[:len_t_list, L//2]), '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                # plt.semilogy(t_list, np.abs(circuit_sz[:,L//2] - mps_sz[:len_t_list, L//2]), '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                plt.setp(axes_list[0].get_xticklabels(), fontsize=8.)

                axes_idx = 1
                if final :
                    axes2.semilogy(t_list, accum_error,'-', color=color, label='$M=%d$' % (depth))
                    axes2.set_ylabel(u'$1 - \prod\mathcal{E}$')
                else:
                    # axes_list[axes_idx].semilogy(t_list, update_error_list,'.', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                    # axes_list[axes_idx].set_ylabel('$1 - \mathcal{E}$')
                    # axes_idx += 1
                    axes_list[axes_idx].semilogy(t_list, accum_error,'-', color=color, label='$M=%d, Niter=%d$' % (depth, N_iter))
                    axes_list[axes_idx].set_ylabel(u'$1 - \prod\mathcal{E}$')
                    axes_idx += 1


                axes_list[axes_idx].plot(t_list[:end_idx], circuit_ent[:end_idx, L//2], '-', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
                axes_list[axes_idx].set_ylabel(u'$ S_{vN}^{L/2} $')
                chi = int(2**(depth))
                axes_list[axes_idx].axhline(y=-np.log(1./chi), color='grey', linewidth=0.5, linestyle='--', label='bound $\\chi=%d$' % chi)
                axes_list[axes_idx].set_ylim([0, np.amax(circuit_ent[:len_t_list, L//2]) + 0.7])

                final_time[-1].append((N_iter, len(t_list)))
            except Exception as e:
                print(e)

    ############################################################
    #### Putting the accumulated error in twin axis instead ####
    ############################################################

    # if not final:
    #     ax2_1 = axes_list[1].twinx()
    #     for depth in range(1,6):
    #         # for idx, N_iter in enumerate([1, 10, 100, 1000]):
    #         for idx, N_iter in enumerate([100000]):
    #             if N_iter < 100:
    #                 continue

    #             color = color_set[1][int((depth-1)*1.4)]
    #             # color = color_set[depth-1][-int(idx*1.5)]

    #             try:
    #                 dir_path = 'data_te/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)

    #                 try:
    #                     tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order)
    #                 except:
    #                     tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order, tmp=False)

    #                 running_idx, my_circuit, E_list, t_list, update_error_list, circuit_sz, circuit_ent, num_iter_array = tuple_loaded

    #                 # filename = 'circuit_depth%d_Niter%d_%s_dt.npy' % (depth, N_iter, order)
    #                 # path = dir_path + filename
    #                 # t_list = np.load(path)

    #                 # filename = 'circuit_depth%d_Niter%d_%s_error.npy' % (depth, N_iter, order)
    #                 # path = dir_path + filename
    #                 # update_error_list = np.load(path)

    #                 update_error_list = np.array(update_error_list)
    #                 accum_error = np.abs(1. - np.multiply.accumulate(1 - update_error_list))
    #                 ax2_1.semilogy(t_list, accum_error,'--', color=color, label='$depth=%d, Niter=%d$' % (depth, N_iter))
    #                 ax2_1.set_ylabel(u'$1 - \prod\mathcal{E}$')
    #             except Exception as e:
    #                 print(e)




    # plt.setp(axes_list[0].get_xticklabels(), fontsize=8)
    plt.setp(axes_list[0].get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0.07)

    # plt.suptitle('Circuit ' + ', $L=$' + str(L) + ', %s-order' % order)
    axes_list[axes_idx].set_xlabel('$Jt$')

    axes_list[0].set_ylim([0.4, 1.0])
    axes_list[0].set_ylabel('$\\langle \sigma_z^{L/2} \\rangle $')
    axes_list[0].set_zorder(100)
    if final:
        axes_list[0].legend(fontsize=7.5, ncol=3, framealpha=1., loc='lower left', bbox_to_anchor=(0., 0.)) #-0.45))
    else:
        axes_list[0].legend(fontsize=8, ncol=3, framealpha=1., loc='lower left', bbox_to_anchor=(0.65, -1.05))

    axes_list[0].set_xlim([0,3.])

    if not final:
        fig.subplots_adjust(left=0.2, top=0.95, bottom=0.1, right=0.93)
    else:
        fig.subplots_adjust(left=0.2, top=0.95, bottom=0.15, right=0.93)

        axes2.axhline(y=1e-3, color='grey', linewidth=0.5, linestyle='--')
        fig2.subplots_adjust(left=0.2, bottom=0.23, top=0.95, right=0.95)
        axes2.set_ylim([1e-8, 1e-1])
        axes2.set_xlabel('$Jt$')

    fig.savefig('figure/time_evolv_%s/circuit_L%d_g%.4f_h%.4f_%s.pdf' % (Hamiltonian, L, g, h, order))
    fig2.savefig('figure/time_evolv_%s/circuit_err_L%d_g%.4f_h%.4f_%s.pdf' % (Hamiltonian, L, g, h, order))


    plt.show()

    # plt.figure()
    # for data in final_time:
    #     print(data)
    #     plt.plot(data, 'x-')

    # plt.show()
