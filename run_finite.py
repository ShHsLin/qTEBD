from scipy import integrate
from scipy.linalg import expm
import numpy as np
import misc, os, sys
import qTEBD


if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    chi = int(sys.argv[3])
    order = str(sys.argv[4])

    assert order in ['1st', '2nd']
    Hamiltonian = 'XXZ'

    ## [TODO] add check whether data already

    J = 1.
    N_iter = 1

    A_list  =  qTEBD.init_mps(L,chi,2)
    H_list  =  qTEBD.get_H(L, J, g, Hamiltonian)
    t_list = [0]
    E_list = [np.sum(qTEBD.expectation_values(A_list, H_list))]
    update_error_list = [0.]
    for dt in [0.05,0.01,0.001]:
        U_list =  qTEBD.make_U(H_list, dt)
        U_half_list =  qTEBD.make_U(H_list, dt/2.)
        for i in range(int(20//dt**(0.75))):
            if order == '2nd':
                Ap_list = qTEBD.apply_U(A_list, U_half_list, 0)
                Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)
                Ap_list = qTEBD.apply_U(Ap_list, U_half_list, 0)
            else:
                Ap_list = qTEBD.apply_U(A_list,  U_list, 0)
                Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)

            # Ap_list = e^(-H) | A_list >
            print("Norm new mps = ", qTEBD.overlap(Ap_list, Ap_list), "new state aimed E = ",
                  np.sum(qTEBD.expectation_values(Ap_list, H_list, check_norm=False))/qTEBD.overlap(Ap_list, Ap_list)
                 )

            for a in range(N_iter):
                A_list  = qTEBD.var_A(A_list, Ap_list, 'right')

            fidelity_reached = np.abs(qTEBD.overlap(Ap_list, A_list))**2 / qTEBD.overlap(Ap_list, Ap_list)
            print("fidelity reached : ", fidelity_reached)
            update_error_list.append(1. - fidelity_reached)
            current_energy = np.sum(qTEBD.expectation_values(A_list, H_list))
            E_list.append(current_energy)
            t_list.append(t_list[-1]+dt)

            print(t_list[-1], E_list[-1])

    dir_path = 'data/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'mps_chi%d_%s_energy.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(E_list))

    filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(t_list))

    filename = 'mps_chi%d_%s_error.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(update_error_list))

    dir_path = 'data/1d_%s_g%.1f/' % (Hamiltonian, g)
    best_E = np.amin(E_list)
    filename = 'mps_chi%d_%s_energy.csv' % (chi, order)
    path = dir_path + filename
    # Try to load file 
    # If data return
    E_dict = {}
    overwrite = True
    try:
        E_array = misc.load_array(path)
        E_dict = misc.nparray_2_dict(E_array)
        assert L in E_dict.keys()
        print("Found data")
        if overwrite:
            raise
    except Exception as error:
        print(error)
        E_dict[L] = best_E
        misc.save_array(path, misc.dict_2_nparray(E_dict))
        # If no data --> generate data
        print("Save new data")




