import matplotlib.pyplot as plt
import numpy as np
import sys, os, misc

if __name__ == '__main__':
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    h = float(sys.argv[3])
    order = str(sys.argv[4])
    Hamiltonian = 'TFI'

    plt.close()
    fig=plt.figure(figsize=(6,9))

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
        pass

    for chi in [2,4,8,32]:
        dir_path = 'data_te/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)

        filename = 'mps_chi%d_%s_energy.npy' % (chi, order)
        path = dir_path + filename
        E_list = np.load(path)

        filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
        path = dir_path + filename
        t_list = np.load(path)

        filename = 'mps_chi%d_%s_error.npy' % (chi, order)
        path = dir_path + filename
        update_error_list = np.load(path)

        filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)
        path = dir_path + filename
        mps_sz = np.load(path)

        filename = 'mps_chi%d_%s_ent_array.npy' % (chi, order)
        path = dir_path + filename
        ent_array = np.load(path)

        ax1 = plt.subplot(311)
        plt.plot(t_list, mps_sz[:,L//2], '-', label='$\\chi=%d$' % chi)
        plt.setp(ax1.get_xticklabels(), fontsize=6)


        ax2 = plt.subplot(312, sharex=ax1)
        plt.semilogy(t_list, update_error_list,'.', label='$\\chi=%d$' % chi)
        plt.ylabel('$1 - \mathcal{F}$')

        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(t_list, ent_array[:,L//2], '.', label='$\\chi=%d$' % chi)
        plt.ylabel('entanglement')
        plt.axhline(y=-np.log(1./chi), color='r', linestyle='-', label='bound $\\chi=%d$' % chi)




    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0)

    plt.suptitle('MPS ' + ', $L=$' + str(L) + ', %s-order' % order)
    plt.xlabel('$\\tau$')
    # plt.legend(['$\\chi=%.0f$'%chi])
    plt.subplot(311)
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    plt.ylabel('$< S_z^{L/2} >$')
    plt.legend()
    # plt.savefig('figure/finite_L%d_chi%d.png' % (L, chi))
    plt.savefig('figure/time_evolv_%s/mps_L%d_g%.4f_h%.4f_%s.png' % (Hamiltonian, L, g, h, order))
    # plt.savefig('figure/finite_L%d.png' % (L))
    plt.show()
