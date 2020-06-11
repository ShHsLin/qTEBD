from scipy import integrate
from scipy.linalg import expm
import numpy as np
import sys
sys.path.append('..')
import misc, os
import qTEBD
import parse_args

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

    args = parse_args.parse_args()

    L = args.L
    Hamiltonian = args.H
    assert Hamiltonian in ['TFI', 'XXZ']

    J = 1.
    if Hamiltonian == 'TFI':
        g = args.g
        h = args.h
        H_list = qTEBD.get_H(Hamiltonian, L, J, g, h)
    elif Hamiltonian == 'XXZ':
        g = delta = args.delta
        H_list = qTEBD.get_H(Hamiltonian, L, J, delta, 0)

    depth = args.depth
    N_iter = args.N_iter
    order = args.order

    assert order in ['1st', '2nd']

    my_circuit = []

    t_list = [0]
    E_list = []
    update_error_list = [0.]

    for dep_idx in range(depth):
        # if dep_idx > 0:
        #     identity_layer = [np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]
        #     my_circuit.append(identity_layer)
        # else:
        #     random_layer = [qTEBD.random_2site_U(2) for i in range(L-1)]
        #     my_circuit.append(random_layer)
        random_layer = []
        for idx in range(L-1):
            if (idx + dep_idx) % 2 == 0:
                random_layer.append(qTEBD.random_2site_U(2))
            else:
                random_layer.append(np.eye(4).reshape([2,2,2,2]))

        my_circuit.append(random_layer)
        current_depth = dep_idx + 1

    product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
    E_list.append(np.sum(qTEBD.expectation_values(mps_of_layer[-1], H_list)))

    for dt in [0.05,0.01,0.001]:
        U_list =  qTEBD.make_U(H_list, dt)
        U_half_list =  qTEBD.make_U(H_list, dt/2.)
        for i in range(int(40//dt**(0.75))):
            mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
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
                mps_of_last_layer, my_circuit = qTEBD.var_circuit2(new_mps, product_state, my_circuit, brickwall=True)

            # [Todo] log the fedility here
            mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
            mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
            assert np.isclose(qTEBD.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
            current_energy = np.sum(qTEBD.expectation_values(mps_of_last_layer, H_list))
            E_list.append(current_energy)
            t_list.append(t_list[-1]+dt)

            print(t_list[-1], E_list[-1])

            fidelity_reached = np.abs(qTEBD.overlap(new_mps, mps_of_last_layer))**2 / qTEBD.overlap(new_mps, new_mps)
            print("fidelity reached : ", fidelity_reached)
            update_error_list.append(1. - fidelity_reached)



    dir_path = 'data_brickwall/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
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

    dir_path = 'data_brickwall/1d_%s_g%.1f/' % (Hamiltonian, g)
    best_E = np.amin(E_list)
    filename = 'circuit_depth%d_Niter%d_%s_energy.csv' % (depth, N_iter, order)
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


