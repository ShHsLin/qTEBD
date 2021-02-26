from scipy import integrate
from scipy.linalg import expm
import numpy as np
import pickle
import os, sys
sys.path.append('..')
import qTEBD, misc

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
    h = float(sys.argv[3])
    depth = int(sys.argv[4])
    N_iter = int(sys.argv[5])
    order = str(sys.argv[6])
    total_t = 5.
    dt = 0.01
    save_each = int(0.1 // dt)
    tol = 1e-12
    cov_crit = tol * 0.1
    max_N_iter = N_iter

    assert order in ['1st', '2nd']
    Hamiltonian = 'TFI'
    H_list  =  qTEBD.get_H(Hamiltonian, L, J, g, h)
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]

    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)

    idx = 0
    my_circuit = []
    t_list = [0]
    E_list = []
    update_error_list = [0.]
    Sz_array = np.zeros([int(total_t // dt) + 1, L], dtype=np.complex)
    ent_array = np.zeros([int(total_t // dt) + 1, L-1], dtype=np.double)
    num_iter_array = np.zeros([int(total_t // dt) + 1], dtype=np.int)


    ################# INITIALIZATION  ######################
    product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    for dep_idx in range(depth):
        identity_layer = [np.eye(4, dtype=np.complex).reshape([2, 2, 2, 2]) for i in range(L-1)]
        my_circuit.append(identity_layer)
        current_depth = dep_idx + 1

    mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
    mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]

    E_list.append(np.sum(mps_func.expectation_values(mps_of_layer[-1], H_list)))
    Sz_array[0, :] = mps_func.expectation_values_1_site(mps_of_layer[-1], Sz_list)
    ent_array[0, :] = mps_func.get_entanglement(mps_of_last_layer)

    dir_path = 'data_te/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)

    ### Find if the result is already obtained ####
    if misc.check_circuit(dir_path, depth, N_iter, order):
        print("Found finished circuit; quit")
        exit()
    else:
        pass


    wf_dir_path = dir_path + 'wf_depth%d_Niter%d_%s/' % (depth, N_iter, order)
    if not os.path.exists(wf_dir_path):
        os.makedirs(wf_dir_path)

    if idx % save_each == 0:
        pickle.dump(my_circuit, open(wf_dir_path + 'T%.1f.pkl' % t_list[-1], 'wb'))

    #### Trying to load the previous unended result ####
    running_idx = len(t_list)
    try:
        tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order)
        running_idx, my_circuit, E_list, t_list, update_error_list, Sz_array, ent_array, num_iter_array = tuple_loaded
        print(" old result loaded ")
        mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
        mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
    except Exception as e:
        print(e)

    stop_crit = 1e-1
    for idx in range(running_idx, int(total_t // dt) + 1):
        # [TODO] remove the assertion below
        assert np.isclose(mps_func.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        if order == '2nd':
            target_mps = qTEBD.apply_U(mps_of_last_layer,  U_half_list, 0)
            target_mps = qTEBD.apply_U(target_mps, U_list, 1)
            target_mps = qTEBD.apply_U(target_mps, U_half_list, 0)
        else:
            # target_mps = [A.copy() for A in mps_of_last_layer]
            # target_mps, _ = mps_func.right_canonicalize(target_mps, no_trunc=True)
            # target_mps, trunc_error = qTEBD.apply_U_all(target_mps,  U_list, False, no_trunc=True)

            target_mps = qTEBD.apply_U(mps_of_last_layer,  U_list, 0)
            target_mps = qTEBD.apply_U(target_mps, U_list, 1)

        mps_func.right_canonicalize(target_mps, no_trunc=True)
        mps_func.left_canonicalize(target_mps, no_trunc=False)

        # target_mps is the e(-H)|psi0> which is not normalizaed.
        target_mps_norm_sq = mps_func.overlap(target_mps, target_mps)
        overlap = mps_func.overlap(mps_of_last_layer, target_mps)
        F = np.abs(overlap) ** 2 / target_mps_norm_sq

        print("Norm new mps = ", target_mps_norm_sq,
              "new state aimed E = ",
              np.sum(mps_func.expectation_values(target_mps, H_list, check_norm=False))/mps_func.overlap(target_mps, target_mps),
              " fidelity_before_opt= ", F
             )

        ###################################
        #### DO one full iteration here  ##
        ###################################
        mps_of_last_layer, my_circuit = qTEBD.var_circuit2(target_mps, product_state,
                                                           my_circuit, brickwall=True)
        # mps_of_last_layer, my_circuit = qTEBD.var_circuit(target_mps, mps_of_last_layer,
        #                                                   my_circuit, product_state)
        overlap = mps_func.overlap(mps_of_last_layer, target_mps)
        F = np.abs(overlap) ** 2 / target_mps_norm_sq
        ###################################

        num_iter = 1
        F_diff = 1
        while (num_iter < max_N_iter and 1-F > tol and F_diff > cov_crit):
            num_iter = num_iter + 1
            mps_of_last_layer, my_circuit = qTEBD.var_circuit2(target_mps, product_state,
                                                               my_circuit, brickwall=True)
            # mps_of_last_layer, my_circuit = qTEBD.var_circuit(target_mps, mps_of_last_layer,
            #                                                   my_circuit, product_state)
            overlap = mps_func.overlap(mps_of_last_layer, target_mps)
            # overlap
            new_F = np.abs(overlap) ** 2 / target_mps_norm_sq
            F_diff = np.abs(new_F - F)
            F = new_F
            if num_iter % 100 == 0:
                print(" at iter = ", num_iter, " F = ", F)

        assert np.isclose(mps_func.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        current_energy = np.sum(mps_func.expectation_values(mps_of_last_layer, H_list))
        E_list.append(current_energy)
        Sz_array[idx, :] = mps_func.expectation_values_1_site(mps_of_last_layer, Sz_list)
        ent_array[idx, :] = mps_func.get_entanglement(mps_of_last_layer)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])

        fidelity_reached = np.abs(mps_func.overlap(target_mps, mps_of_last_layer))**2 / mps_func.overlap(target_mps, target_mps)
        print("fidelity reached : ", fidelity_reached)
        update_error_list.append(1. - fidelity_reached)
        num_iter_array[idx] = num_iter

        if idx % save_each == 0:
            pickle.dump(my_circuit, open(wf_dir_path + 'T%.1f.pkl' % t_list[-1], 'wb'))

            misc.save_circuit(dir_path, depth, N_iter, order,
                              my_circuit, E_list, t_list, update_error_list,
                              Sz_array, ent_array, num_iter_array, True)

        ################
        ## Forcing to stop if truncation is already too high.
        ################
        total_trunc_error = np.abs(1- np.multiply.reduce(1. - np.array(update_error_list)))
        print('----------- total_trunc_error = ', total_trunc_error, '------------')
        if total_trunc_error > stop_crit:
            break

    num_data = len(t_list)
    Sz_array = Sz_array[:num_data, :]
    ent_array = ent_array[:num_data, :]
    num_iter_array = num_iter_array[:num_data]



    misc.save_circuit(dir_path, depth, N_iter, order,
                      my_circuit, E_list, t_list, update_error_list,
                      Sz_array, ent_array, num_iter_array, False)

