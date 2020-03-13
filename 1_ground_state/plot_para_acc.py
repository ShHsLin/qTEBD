# import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import sys, os, misc
import seaborn as sns
# sns.set()

if __name__ == '__main__':
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    order = str(sys.argv[3])
    Hamiltonian = 'XXZ'

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

    circuit_num_para = []
    circuit_dE = []

    for depth in [1, 2, 3, 4]:
        E_list = []
        for N_iter in [1, 2, 10]:
            try:
                dir_path = 'data/1d_%s_g%.1f/' % (Hamiltonian, g)
                filename = 'circuit_depth%d_Niter%d_%s_energy.csv' % (depth, N_iter, order)
                path = dir_path + filename
                # Try to load file 
                # If data return
                circuit_E_array = misc.load_array(path)
                circuit_E_dict = misc.nparray_2_dict(circuit_E_array)
                circuit_E = circuit_E_dict[L]
                print("Found circuit data")
                E_list.append(circuit_E)
            except:
                pass

        circuit_E = np.amin(E_list)
        circuit_num_para.append( depth * 16 * (L-1) )
        circuit_dE.append( np.abs((circuit_E - exact_E)/exact_E) )


            # ax2 = plt.subplot(212, sharex=ax1)
            # plt.semilogy(t_list, update_error_list,'.', label='depth$=%d$, Niter$=%d$' % (depth, N_iter))
            # plt.ylabel('$1 - \mathcal{F}$')

    plt.loglog(circuit_dE, circuit_num_para, 'o--', label='circuit')
    plt.setp(plt.gca().get_xticklabels(), fontsize=6)
    plt.ylabel('num para')
    plt.xlabel('$E_{\\tau} - E_0$')


    dmrg_num_para = []
    dmrg_dE = []
    for chi in [2, 4, 8]:
        dir_path = 'data/1d_%s_g%.1f/' % (Hamiltonian, g)
        filename = 'dmrg_chi%d_energy.csv' % chi
        path = dir_path + filename
        dmrg_E_array = misc.load_array(path)
        dmrg_E_dict = misc.nparray_2_dict(dmrg_E_array)
        dmrg_E = dmrg_E_dict[L]
        print("Found dmrg data")
        dmrg_num_para.append( 2 * L * chi ** 2 )
        dmrg_dE.append( np.abs((dmrg_E - exact_E)/exact_E))

    plt.loglog(dmrg_dE, dmrg_num_para, 'x-', color='r', label='dmrg')
    plt.gca().set_xlim(1e-1, 1e-6)
    plt.gca().invert_xaxis()

    # pl.title('2-site gate circuit' + ' $depth=%d$' % depth + ', $L=$' + str(L))
    # plt.suptitle('Circuit ' + ', $L=$' + str(L) + ' %s-order' % order)
    # plt.xlabel('$\\tau$')
    plt.legend()
    plt.savefig('figure/%s/scaling_L%d_g%.1f_%s.png' % (Hamiltonian, L, g, order))

    plt.show()
