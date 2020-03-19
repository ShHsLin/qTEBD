import matplotlib.pyplot as plt
import numpy as np
import sys, os
import pickle as pkl
sys.path.append('..')
import misc
import qTEBD
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
    product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]

    plt.close()
    fig=plt.figure(figsize=(6,8))

    ############################################
    circuit_dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
    mps_dir_path = 'data_tebd/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)

    final_time = []
    for depth in range(1,5):
        final_time.append([])
        for idx, N_iter in enumerate([1,10,100,]):
            if depth==1 and N_iter > 1:
                continue

            color = color_set[depth-1][int(idx*1.5)]

            circuit_sub_dir = '/wf_depth%d_Niter%d_%s/' % (depth, N_iter, order)
            mps_sub_dir = '/wf_chi%d_%s/' % (2**depth, order)
            fidelity_list = []
            for t_idx in range(300):
                try:
                    T = 0.1 * t_idx
                    circuit_path = circuit_dir_path + circuit_sub_dir + 'T%.1f.pkl' % T
                    circuit = pkl.load(open(circuit_path, 'rb'))
                    mps_path = mps_dir_path + mps_sub_dir + 'T%.1f.pkl' % T
                    mps = pkl.load(open(mps_path, 'rb'))

                    mps_of_layer = qTEBD.circuit_2_mps(circuit, product_state)
                    overlap = qTEBD.overlap(mps_of_layer[-1], mps)
                    F = np.abs(overlap) ** 2
                    print(" F(", t_idx, ") = ", F)
                    fidelity_list.append(F)
                except Exception as e:
                    print(e)
                    break

            plt.plot(1.-np.array(fidelity_list)+1e-15, color=color, label='dep=%d, Niter=%d' % (depth, N_iter))
            plt.gca().set_yscale('log')

            # plt.plot(np.array(fidelity_list), color=color, label='dep=%d, Niter=%d' % (depth, N_iter))


    plt.legend()
    plt.title('err_fidelity_log')
    plt.savefig('figure/time_evolv_TFI/fidelity.png')
    plt.show()


