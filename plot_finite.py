# import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import sys, os, misc

if __name__ == '__main__':
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    H = 'XXZ'

    plt.close()
    fig=plt.figure(figsize=(6,8))

    ############################################
    dir_path = 'data/1d_%s_g%.1f/' % (Hamiltonian, g)
    filename = 'exact_energy.csv'
    path = dir_path + filename
    # Try to load file 
    # If data return
    exact_E_array = misc.load_array(path)
    exact_E_dict = misc.nparray_2_dict(exact_E_array)
    exact_E = exact_E_dict[L]
    print("Found exact energy data")


    order = '2nd'
    for chi in [2, 4]:
        dir_path = 'data/1d_%s_g%.1f/' % (Hamiltonian, g)
        filename = 'mps_chi%d_%s_energy.csv' % (chi, order)
        path = dir_path + filename
        mps_E_array = misc.load_array(path)
        mps_E_dict = misc.nparray_2_dict(mps_E_array)
        mps_E = mps_E_dict[L]
        print("Found mps data")

        filename = 'dmrg_chi%d_energy.csv' % chi
        path = dir_path + filename
        dmrg_E_array = misc.load_array(path)
        dmrg_E_dict = misc.nparray_2_dict(dmrg_E_array)
        dmrg_E = dmrg_E_dict[L]
        print("Found dmrg data")




        ############################################
        dir_path = 'data/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)

        filename = 'mps_chi%d_%s_energy.npy' % (chi, order)
        path = dir_path + filename
        E_list = np.load(path)

        filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
        path = dir_path + filename
        t_list = np.load(path)

        filename = 'mps_chi%d_%s_error.npy' % (chi, order)
        path = dir_path + filename
        update_error_list = np.load(path)
        delta_list = E_list - exact_E


        ax1 = plt.subplot(211)
        plt.semilogy(t_list, delta_list,'.', label='$\\chi=%d$' % chi)
        plt.axhline(y=dmrg_E-exact_E, color='r', linestyle='-', label='dmrg $\\chi=%d$' % chi)
        plt.setp(ax1.get_xticklabels(), fontsize=6)
        plt.ylabel('$E_{\\tau} - E_0$')


        ax2 = plt.subplot(212, sharex=ax1)
        plt.semilogy(t_list, update_error_list,'.', label='$\\chi=%d$' % chi)
        plt.ylabel('$1 - \mathcal{F}$')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0)

    plt.suptitle('MPS ' + ', $L=$' + str(L) + ', %s-order' % order)
    plt.xlabel('$\\tau$')
    # plt.legend(['$\\chi=%.0f$'%chi])
    plt.legend()
    # plt.savefig('figure/finite_L%d_chi%d.png' % (L, chi))
    plt.savefig('figure/%s/finite_L%d_g%.1f_%s.png' % (Hamiltonian, L, g, order))
    # plt.savefig('figure/finite_L%d.png' % (L))
    plt.show()
