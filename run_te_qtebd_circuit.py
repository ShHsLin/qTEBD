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
    N_iter = int(sys.argv[4])
    order = str(sys.argv[5])
    total_t = 3.

    assert order in ['1st', '2nd']
    Hamiltonian = 'TFI'
    H_list  =  qTEBD.get_H(L, J, g, Hamiltonian)
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]


    my_circuit = []

    t_list = [0]
    E_list = []
    update_error_list = [0.]
    dt = 0.01
    Sz_array = np.zeros([int(total_t // dt) + 1, L])

    for dep_idx in range(depth):
        # if dep_idx > 0:
        #     identity_layer = [np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]
        #     my_circuit.append(identity_layer)
        # else:
        #     random_layer = [qTEBD.random_2site_U(2) for i in range(L-1)]
        #     my_circuit.append(random_layer)

        # random_layer = [qTEBD.random_2site_U(2) for i in range(L-1)]
        # my_circuit.append(random_layer)
        identity_layer = [np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]
        my_circuit.append(identity_layer)
        current_depth = dep_idx + 1

    mps_of_layer = qTEBD.circuit_2_mps(my_circuit)
    E_list.append(np.sum(qTEBD.expectation_values(mps_of_layer[-1], H_list)))
    Sz_array[0, :] = qTEBD.expectation_values_1_site(mps_of_layer[-1], Sz_list)

    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)
    for idx in range(1, int(total_t // dt) + 1):
        mps_of_layer = qTEBD.circuit_2_mps(my_circuit)
        mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
        # [TODO] remove the assertion below
        assert np.isclose(qTEBD.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        if order == '2nd':
            new_mps = qTEBD.apply_U(mps_of_last_layer,  U_half_list, 0)
            new_mps = qTEBD.apply_U(new_mps, U_list, 1)
            new_mps = qTEBD.apply_U(new_mps, U_half_list, 0)
        else:
            new_mps = qTEBD.apply_U(mps_of_last_layer,  U_list, 0)
            new_mps = qTEBD.apply_U(new_mps, U_list, 1)

        print("Norm new mps = ", qTEBD.overlap(new_mps, new_mps), "new state aimed E = ",
              np.sum(qTEBD.expectation_values(new_mps, H_list, check_norm=False))/qTEBD.overlap(new_mps, new_mps)
             )
        # new_mps is the e(-H)|psi0> which is not normalizaed.

        for iter_idx in range(N_iter):
            iter_mps = [A.copy() for A in new_mps]
            for var_dep_idx in range(current_depth, 0, -1):
            # for var_dep_idx in range(current_depth, current_depth-1, -1):
                # circuit is modified inplace
                # new mps is returned
                iter_mps, new_layer = qTEBD.var_layer([A.copy() for A in iter_mps],
                                                      my_circuit[var_dep_idx - 1],
                                                      mps_of_layer[var_dep_idx - 1],
                                                     )
                assert(len(new_layer) == L -1)
                my_circuit[var_dep_idx - 1] = new_layer

            mps_of_layer = qTEBD.circuit_2_mps(my_circuit)

        # [Todo] log the fedility here
        mps_of_layer = qTEBD.circuit_2_mps(my_circuit)
        mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
        assert np.isclose(qTEBD.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        current_energy = np.sum(qTEBD.expectation_values(mps_of_last_layer, H_list))
        E_list.append(current_energy)
        Sz_array[idx, :] = qTEBD.expectation_values_1_site(mps_of_last_layer, Sz_list)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])

        fidelity_reached = np.abs(qTEBD.overlap(new_mps, mps_of_last_layer))**2 / qTEBD.overlap(new_mps, new_mps)
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

