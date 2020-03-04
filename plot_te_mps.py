import matplotlib.pyplot as plt
import numpy as np
import sys, os, misc

if __name__ == '__main__':
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    order = str(sys.argv[3])
    Hamiltonian = 'TFI'

    plt.close()
    fig=plt.figure(figsize=(6,8))

    ############################################
    dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
    filename = 'ed_sz_array.npy'
    exact_sz = np.load(dir_path + filename)
    filename = 'ed_dt_array.npy'
    exact_dt = np.load(dir_path + filename)

    ax1 = plt.subplot(211)
    plt.plot(exact_dt[:], exact_sz[:,L//2],'--k', label='exact')
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    plt.ylabel('$< S_z^{L/2} >$')

    for chi in [2, 4, 8]:
        dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
        filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)
        path = dir_path + filename
        mps_sz = np.load(path)

        ############################################
        dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)

        filename = 'mps_chi%d_%s_energy.npy' % (chi, order)
        path = dir_path + filename
        E_list = np.load(path)

        filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
        path = dir_path + filename
        t_list = np.load(path)

        filename = 'mps_chi%d_%s_error.npy' % (chi, order)
        path = dir_path + filename
        update_error_list = np.load(path)


        ax1 = plt.subplot(211)
        plt.plot(t_list, mps_sz[:,L//2], '-', label='$\\chi=%d$' % chi)
        plt.setp(ax1.get_xticklabels(), fontsize=6)


        ax2 = plt.subplot(212, sharex=ax1)
        plt.semilogy(t_list, update_error_list,'.', label='$\\chi=%d$' % chi)
        plt.ylabel('$1 - \mathcal{F}$')




    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0)

    plt.suptitle('MPS ' + ', $L=$' + str(L) + ', %s-order' % order)
    plt.xlabel('$\\tau$')
    # plt.legend(['$\\chi=%.0f$'%chi])
    plt.subplot(211)
    plt.legend()
    # plt.savefig('figure/finite_L%d_chi%d.png' % (L, chi))
    plt.savefig('figure/time_evolv_%s/mps_L%d_g%.1f_%s.png' % (Hamiltonian, L, g, order))
    # plt.savefig('figure/finite_L%d.png' % (L))
    plt.show()
