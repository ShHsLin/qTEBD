import numpy as np
import pickle
import os, sys
sys.path.append('..')
import qTEBD, misc
import parse_args
import mps_func

'''
Simple example to use the circuit to approximate

T = 3*dt

U^e(dt/2) U^o(dt) U^e(dt) U^o(dt) U^e(dt) U^o(dt) U^e(dt/2)

To run the code:
    python run_optimization_simpleU.py --L 10 --N_iter 100 --depth 3 --T 0.2

'''

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)

    args = parse_args.parse_args()

    L = args.L
    J = 1.
    depth = args.depth
    N_iter = args.N_iter
    T = args.T


    tol = 1e-12
    cov_crit = tol * 0.1
    max_N_iter = N_iter


    ############### SET/LOAD TARGET OPERATOR ######################

    ## Provide filename in variable "filename"
    ## Load the operator by

    # target_mpo = pickle.load(open(filename, 'rb'))

    Hamiltonian = 'TFI'
    H_list  =  qTEBD.get_H(Hamiltonian, L, J, g=1.2, h=0.1)
    # suppose we use 7 layers of second order trotterization
    dt = T / 3.
    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)
    U_half_even = [U_half_list[i].copy() if i % 2 == 0
                   else np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]
    U_half_odd = [U_half_list[i].copy() if i % 2 == 1
                  else np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]
    U_even = [U_list[i].copy() if i % 2 == 0
              else np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]
    U_odd = [U_list[i].copy() if i % 2 == 1
             else np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]

    op_circuit = [U_half_even, U_odd, U_even, U_odd, U_even, U_odd, U_half_even]
    mpo_base = [np.eye(2).reshape([2, 1, 1, 2]) for i in range(L)]

    mpo_operator_list = qTEBD.circuit_2_mpo(op_circuit, mpo_base)
    mpo_operator = mpo_operator_list[-1]
    full_operator = mps_func.MPO_2_operator(mpo_operator)

    # taking the complex conjugate
    target_mpo = [t.transpose([3, 1, 2, 0]).conj() for t in mpo_operator]


    ############### SET UP INITIALIZATION #################
    ## We should also try out identity initialization and
    ## trotterization initialization

    idx = 0
    my_circuit = []
    t_list = [0]
    error_list = []
    num_iter_array = np.zeros([N_iter], dtype=np.int)


    ################# INITIALIZATION  ######################
    product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    for dep_idx in range(depth):
        my_circuit.append([qTEBD.random_2site_U(2) for i in range(L-1)])
        current_depth = dep_idx + 1


    result_mpo_list = qTEBD.circuit_2_mpo(my_circuit, mpo_base)
    result_mpo = result_mpo_list[-1]
    full_result_operator = mps_func.MPO_2_operator(result_mpo)
    err = np.linalg.norm(full_result_operator - full_operator) ** 2

    print("err before optimization : ", err)

    error_list.append(err)


    stop_crit = 1e-4
    for idx in range(0, N_iter):
        #################################
        #### variational optimzation ####
        #################################
        mpo_of_last_layer, my_circuit = qTEBD.var_circuit_mpo(target_mpo,
                                                              my_circuit,)

        #################
        #### Measure ####
        #################
        result_mpo_list = qTEBD.circuit_2_mpo(my_circuit, mpo_base)
        result_mpo = result_mpo_list[-1]
        full_result_operator = mps_func.MPO_2_operator(result_mpo)
        err = np.linalg.norm(full_result_operator - full_operator) ** 2

        print("idx = ", idx, "err = ", err)

        error_list.append(err)
        num_iter_array[idx] = idx
        t_list.append(idx)

        ################
        ## Forcing to stop if already converge
        ################
        if (err < 1e-12 or np.abs((error_list[-1] - error_list[-2])/error_list[-1]) < stop_crit):
            break

    num_data = len(error_list)
    num_iter_array = num_iter_array[:num_data]



