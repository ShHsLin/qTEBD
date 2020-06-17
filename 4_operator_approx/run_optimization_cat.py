import numpy as np
import pickle
import os, sys
sys.path.append('..')
import qTEBD, misc
import parse_args
import mps_func

'''
Simple example to use the circuit to approximate
| cat > < all up |
'''

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)

    args = parse_args.parse_args()

    L = args.L
    J = 1.
    depth = args.depth
    N_iter = args.N_iter


    save_each = 100
    tol = 1e-12
    cov_crit = tol * 0.1
    max_N_iter = N_iter


    ############### LOAD TARGET STATE ######################

    ## Provide filename in variable "filename"
    ## Load the state by

    # target_mps = pickle.load(open(filename, 'rb'))

    cat_state = np.zeros([2**L])
    cat_state[0] = cat_state[-1] = 1./np.sqrt(2)
    target_mps = mps_func.state_2_MPS(cat_state, L, chimax=100)

    mpo_operator = [np.einsum('pjk,q->pjkq', t, np.array([1, 0])) for t in target_mps]
    # taking the complex conjugate
    target_mpo = [t.transpose([3, 1, 2, 0]).conj() for t in mpo_operator]


    ############### SET UP INITIALIZATION #################

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

    mps_of_layer = qTEBD.circuit_2_mps(my_circuit, product_state)
    mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]

    fidelity_reached = np.abs(qTEBD.overlap(target_mps, mps_of_last_layer))**2
    error_list.append(1. - fidelity_reached)
    print("fidelity before optimization : ", fidelity_reached)


    stop_crit = 1e-1
    for idx in range(0, N_iter):
        #################################
        #### variational optimzation ####
        #################################
        mpo_of_last_layer, my_circuit = qTEBD.var_circuit_mpo(target_mpo,
                                                              my_circuit,)

        density_mat = np.einsum('pabq,PbcQ,tcdr->pPtqQr', mpo_of_last_layer[0], mpo_of_last_layer[1], mpo_of_last_layer[2]).reshape([8, 8])

        our_mps, trunc_err = qTEBD.apply_U_all(product_state, my_circuit[0])
        print("trunc_err = ", trunc_err)
        print("cat state = ", cat_state)
        print("our state = ", mps_func.MPS_2_state(our_mps))

        #################
        #### Measure ####
        #################

        fidelity_reached = np.abs(qTEBD.overlap(target_mps, our_mps))**2

        print("fidelity reached : ", fidelity_reached)
        error_list.append(1. - fidelity_reached)
        num_iter_array[idx] = idx

        ################
        ## Forcing to stop if already converge
        ################
        if (fidelity_reached > 1 - 1e-12 or np.abs((error_list[-1] - error_list[-2])/error_list[-1]) < 1e-4) and idx > save_each:
            break

    num_data = len(error_list)
    num_iter_array = num_iter_array[:num_data]


