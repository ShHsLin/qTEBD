from scipy import integrate
from scipy.linalg import expm
import numpy as np
import misc, os, sys
import qTEBD

'''
    Algorithm:
        (1.) first call circuit_2_mps to get the list of mps-reprentation of
        circuit up to each layer.
        (2.1) collapse imaginary time evolution on layer-n circuit e^(-H) |n> = |psi>
        (2.2) normalize |psi>
        (3.) var optimize layer-n by maximizing < psi | U(n) | n-1>
        (4.) collapse layer-n optimized on |psi> getting new |psi>
        (5.) var optimize layer-n-1 by maximizing < psi | U(n-1) | n-2 >
        [ToDo] check the index n above whether is consistent with the code.
        ...
'''

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
    L = int(sys.argv[1])
    J = 1.
    g = float(sys.argv[2])
    depth = int(sys.argv[3])
    max_N_iter = int(sys.argv[4])
    order = str(sys.argv[5])
    total_t = 15.
    dt = 0.01
    tol = 1e-8
    cov_crit = tol * 0.1
    # max_N_iter = 200

    assert order in ['1st', '2nd']
    Hamiltonian = 'TFI'
    H_list  =  qTEBD.get_H(L, J, g, Hamiltonian)
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]

    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)

    my_circuit = []
    t_list = [0]
    E_list = []
    update_error_list = [0.]
    Sz_array = np.zeros([int(total_t // dt) + 1, L], dtype=np.complex)

    for dep_idx in range(depth):
        identity_layer = [np.eye(4, dtype=np.complex).reshape([2, 2, 2, 2]) for i in range(L-1)]
        my_circuit.append(identity_layer)
        current_depth = dep_idx + 1

    mps_of_layer, full_cache = qTEBD.circuit_2_mps(my_circuit)
    mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]

    E_list.append(np.sum(qTEBD.expectation_values(mps_of_layer[-1], H_list)))
    Sz_array[0, :] = qTEBD.expectation_values_1_site(mps_of_layer[-1], Sz_list)

    for idx in range(1, int(total_t // dt) + 1):
        # [TODO] remove the assertion below
        assert np.isclose(qTEBD.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        if order == '2nd':
            target_mps = qTEBD.apply_U(mps_of_last_layer,  U_half_list, 0)
            target_mps = qTEBD.apply_U(target_mps, U_list, 1)
            target_mps = qTEBD.apply_U(target_mps, U_half_list, 0)
        else:
            target_mps = qTEBD.apply_U(mps_of_last_layer,  U_list, 0)
            target_mps = qTEBD.apply_U(target_mps, U_list, 1)

        if idx == 1:
            my_circuit[0] = [U.copy() for U in U_list]
            continue


        # target_mps is the e(-H)|psi0> which is not normalizaed.
        target_mps_norm_sq = qTEBD.overlap(target_mps, target_mps)
        overlap = qTEBD.overlap(mps_of_last_layer, target_mps)
        F = np.abs(overlap) ** 2 / target_mps_norm_sq

        print("Norm new mps = ", target_mps_norm_sq,
              "new state aimed E = ",
              np.sum(qTEBD.expectation_values(target_mps, H_list, check_norm=False))/qTEBD.overlap(target_mps, target_mps),
              " fidelity_before_opt= ", F
             )

        ###################################
        #### DO one full iteration here  ##
        ###################################
        mps_of_last_layer, full_cache, my_circuit = qTEBD.var_circuit(target_mps, full_cache, my_circuit)
        overlap = qTEBD.overlap(mps_of_last_layer, target_mps)
        F = np.abs(overlap) ** 2 / target_mps_norm_sq
        ###################################

        num_iter = 0
        F_diff = 1
        while (num_iter < max_N_iter and 1-F > tol and F_diff > cov_crit):
            num_iter = num_iter + 1
            iter_mps = [A.copy() for A in target_mps]
            mps_of_last_layer, full_cache, my_circuit = qTEBD.var_circuit(target_mps, full_cache, my_circuit)
            overlap = qTEBD.overlap(mps_of_last_layer, target_mps)
            # overlap
            new_F = np.abs(overlap) ** 2 / target_mps_norm_sq
            F_diff = np.abs(new_F - F)
            F = new_F
            print(" at iter = ", num_iter, " F = ", F)
            # mps_of_layer = qTEBD.circuit_2_mps(my_circuit)

        # mps_of_layer = qTEBD.circuit_2_mps(my_circuit)
        # mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
        assert np.isclose(qTEBD.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        current_energy = np.sum(qTEBD.expectation_values(mps_of_last_layer, H_list))
        E_list.append(current_energy)
        Sz_array[idx, :] = qTEBD.expectation_values_1_site(mps_of_last_layer, Sz_list)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])

        fidelity_reached = np.abs(qTEBD.overlap(target_mps, mps_of_last_layer))**2 / qTEBD.overlap(target_mps, target_mps)
        print("fidelity reached : ", fidelity_reached)
        update_error_list.append(1. - fidelity_reached)


    dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'circuit_depth%d_Niter%d_%s_energy.npy' % (depth, N_iter, order)
    path = dir_path + filename
    np.save(path, np.array(E_list))

    filename = 'circuit_depth%d_Niter%d_%s_dt.npy' % (depth, N_iter, order)
    path = dir_path + filename
    np.save(path, np.array(t_list))

    filename = 'circuit_depth%d_Niter%d_%s_error.npy' % (depth, N_iter, order)
    path = dir_path + filename
    np.save(path, np.array(update_error_list))

    filename = 'circuit_depth%d_Niter%d_%s_sz_array.npy' % (depth, N_iter, order)
    path = dir_path + filename
    np.save(path, Sz_array)

