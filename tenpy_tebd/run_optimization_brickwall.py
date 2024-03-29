import numpy as np
import pickle
import os, sys
sys.path.append('..')
import qTEBD, misc

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
    ## the target state is corresponding to time T.
    T = float(sys.argv[7])


    save_each = 100
    tol = 1e-12
    cov_crit = tol * 0.1
    max_N_iter = N_iter

    assert order in ['1st', '2nd', '4th']

    Hamiltonian = 'TFI'
    H_list  =  qTEBD.get_H(Hamiltonian, L, J, g, h)
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]


    ############### LOAD TARGET STATE ######################
    if np.isclose(g, 1.):
        chi = 128
    else:
        chi = 32

    # data_tebd_dt1.000000e-03/1d_TFI_g1.4000_h0.*/L31/trunc_wf_chi32_4th/
    mps_dir_path = './data_tebd_dt%e/1d_%s_g%.4f_h%.4f/L%d/trunc_wf_chi%d_4th/' % (1e-3, Hamiltonian, g, h, L, chi)
    filename = mps_dir_path + 'T%.1f.pkl' % T
    target_mps = pickle.load(open(filename, 'rb'))

    ############### SET UP INITIALIZATION #################
    ## We should test identity initialization and
    ## trotterization initialization
    dt = T / (depth /2.)
    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)

    idx = 0
    my_circuit = []
    E_list = []
    t_list = [0]
    error_list = []
    Sz_array = np.zeros([N_iter, L], dtype=np.complex)
    ent_array = np.zeros([N_iter, L-1], dtype=np.double)
    num_iter_array = np.zeros([N_iter], dtype=np.int)


    ################# INITIALIZATION  ######################
    product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    for dep_idx in range(depth):
        ## Trotterization initization
        # my_circuit.append([t.copy() for t in U_list])
        # current_depth = dep_idx + 1

        random_layer = []
        for idx in range(L-1):
            if (idx + dep_idx) % 2 == 0:
                # random_layer.append(qTEBD.random_2site_U(2))
                random_layer.append(U_list[idx].copy())
            else:
                random_layer.append(np.eye(4).reshape([2,2,2,2]))

        my_circuit.append(random_layer)
        current_depth = dep_idx + 1


    mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
    mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]

    E_list.append(np.sum(mps_func.expectation_values(mps_of_layer[-1], H_list)))
    Sz_array[0, :] = mps_func.expectation_values_1_site(mps_of_layer[-1], Sz_list)
    ent_array[0, :] = mps_func.get_entanglement(mps_of_last_layer)
    fidelity_reached = np.abs(mps_func.overlap(target_mps, mps_of_last_layer))**2
    error_list.append(1. - fidelity_reached)


    dir_path = 'data_brickwall/1d_%s_g%.4f_h%.4f/L%d_chi%d/T%.1f/' % (Hamiltonian, g, h, L, chi, T)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    running_idx = len(E_list)
    if misc.check_circuit(dir_path, depth, N_iter, order):
        print("Found finished circuit; quit")
        exit()
    else:
        pass

    try:
        tuple_loaded = misc.load_circuit(dir_path, depth, N_iter, order)
        running_idx, my_circuit, E_list, t_list, error_list, Sz_array, ent_array, num_iter_array = tuple_loaded
        print(" old result loaded ")
        mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
        mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
    except Exception as e:
        print(e)

    stop_crit = 1e-1
    assert np.isclose(mps_func.overlap(target_mps, target_mps), 1.)
    for idx in range(running_idx, N_iter):
        #################################
        #### variational optimzation ####
        #################################

        try:
            mps_of_last_layer, my_circuit = qTEBD.var_circuit(target_mps, mps_of_last_layer,
                                                              my_circuit, product_state, brickwall=True)
        except:
            mps_of_last_layer, my_circuit = qTEBD.var_circuit2(target_mps, product_state, my_circuit,
                                                               brickwall=True
                                                              )
        #################
        #### Measure ####
        #################
        assert np.isclose(mps_func.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        current_energy = np.sum(mps_func.expectation_values(mps_of_last_layer, H_list))
        E_list.append(current_energy)
        Sz_array[idx, :] = mps_func.expectation_values_1_site(mps_of_last_layer, Sz_list)
        ent_array[idx, :] = mps_func.get_entanglement(mps_of_last_layer)
        fidelity_reached = np.abs(mps_func.overlap(target_mps, mps_of_last_layer))**2

        print("iter=", idx, " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])
        print("fidelity reached : ", fidelity_reached)
        error_list.append(1. - fidelity_reached)
        num_iter_array[idx] = idx
        t_list.append(idx)

        if idx % save_each == 0:
            misc.save_circuit(dir_path, depth, N_iter, order,
                              my_circuit, E_list, t_list, error_list,
                              Sz_array, ent_array, num_iter_array, True)

        ################
        ## Forcing to stop if already converge
        ################
        if (fidelity_reached > 1 - 1e-12 or np.abs((error_list[-1] - error_list[-2])/error_list[-1]) < 1e-4) and idx > save_each:
            break

    num_data = len(E_list)
    Sz_array = Sz_array[:num_data, :]
    ent_array = ent_array[:num_data, :]
    num_iter_array = num_iter_array[:num_data]



    misc.save_circuit(dir_path, depth, N_iter, order,
                      my_circuit, E_list, t_list, error_list,
                      Sz_array, ent_array, num_iter_array, False)

