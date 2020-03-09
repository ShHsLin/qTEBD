import matplotlib.pyplot as plt
import numpy as np
import sys, os, misc

if __name__ == '__main__':
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    order = str(sys.argv[3])
    Hamiltonian = 'TFI'

    plt.close()
    fig=plt.figure(figsize=(6,9))

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
        pass

    for depth, N_iter in [(1, 1), (2, 10)]:
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
        plt.plot(t_list, circuit_sz[:,L//2], '-', label='$depth=%d, Niter=%d$' % (depth, N_iter))
        plt.setp(ax1.get_xticklabels(), fontsize=6)


        ax2 = plt.subplot(312, sharex=ax1)
        plt.semilogy(t_list, update_error_list,'.', label='$depth=%d, Niter=%d$' % (depth, N_iter))
        plt.ylabel('$1 - \mathcal{F}$')

        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(t_list, circuit_ent[:,L//2], '.', label='$depth=%d, Niter=%d$' % (depth, N_iter))
        plt.ylabel('engtanglement')
        chi = int(np.log2(depth))
        #plt.axhline(y=-np.log(1./chi), color='r', linestyle='-', label='bound $\\chi=%d$' % chi)



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
