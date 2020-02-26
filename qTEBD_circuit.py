from scipy import integrate
from scipy.linalg import expm
import numpy as np
import pylab as pl
import exact_diagonalization as ed
import misc, os
import sys

def get_H_TFI(L, J, g):
    '''
    Return:
        H: list of local hamiltonian
    '''
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    id = np.eye(2)
    d = 2

    def h(gl, gr, J):
        # return (-np.kron(sz, sz) * J - gr * np.kron(id, sx) - gl * np.kron(sx, id)).reshape([d] * 4)
        return (-np.kron(sx, sx) * J - gr * np.kron(id, sz) - gl * np.kron(sz, id)).reshape([d] * 4)

    H = []
    for j in range(L - 1):
        if j == 0:
            gl = g
        else:
            gl = 0.5 * g
        if j == L - 2:
            gr = 1. * g
        else:
            gr = 0.5 * g
        H.append(h(gl, gr, J))
    return H

def make_U(H, t):
    """ U = exp(-t H) """
    d = H[0].shape[0]
    return [expm(-t * h.reshape((d**2, -1))).reshape([d] * 4) for h in H]

def init_mps(L, chi, d):
    A_list = []
    for i in range(L):
        chi1 = np.min([d**np.min([i,L-i]),chi])
        chi2 = np.min([d**np.min([i+1,L-i-1]),chi])
        A_list.append(polar(0.5 - np.random.rand(d,chi1,chi2)))

    return A_list

def polar(A):
    d,chi1,chi2 = A.shape
    Y,D,Z = np.linalg.svd(A.reshape(d*chi1,chi2), full_matrices=False)
    return np.dot(Y,Z).reshape([d,chi1,chi2])

def random_2site_U(d, factor=1e-2):
    A = np.random.rand(d**2, d**2) * factor
    A = A-A.T
    U = (np.eye(d**2)-A).dot(np.linalg.inv(np.eye(d**2)+A))
    return U.reshape([d] * 4)
    # M = np.random.rand(d ** 2, d ** 2)
    # Q, _ = np.linalg.qr(0.5 - M)
    # return Q.reshape([d] * 4)

def circuit_2_mps(circuit, chi=None):
    '''
    Input:
        circuit is a list of list of U, i.e.
        [[U0(t0), U1(t0), U2(t0), ...], [U0(t1), U1(t1), ...], ...]
        circuit[0] corresponds to layer-0,
        circuit[1] corresponds to layer-1, and so on.

    Goal:
        We compute the mps representaion of each layer
        without truncation.
        The mps is not in necessary in canonical form.
        But since we do not make any truncation, that
        should be fine.

        [ToDo] : add a truncation according to chi?

    return:
        mps representation of each layer
        mps_of_layer[0] gives the product state |psi0>
        mps_of_layer[1] gives the U(0) |psi0>
    '''
    depth = len(circuit)
    L = len(circuit[0]) + 1
    A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]

    mps_of_layer = []
    mps_of_layer.append([A.copy() for A in A_list])
    # A_list is modified inplace always.
    # so we always have to copy A tensors.
    for dep_idx in range(depth):
        U_list = circuit[dep_idx]
        A_list = apply_U_all(A_list, U_list)
        mps_of_layer.append([A.copy() for A in A_list])

    return mps_of_layer

def var_layer(new_mps, layer_gate, mps_old):
    '''
    max
    <new_mps | layer_gate | mps_old>
    |    |    |    |
    -------------------- layer 0

    -------------------- layer 1
    .
    .
    .
    -------------------- layer n

    -------------------- Imaginary time evolution

    -------------------- layer n
    .
    .
    .
    -------------------- layer 1

    -------------------- layer 0
    |    |    |    |

    =

    -------------------- mps-representation of all layer-(n) circuit
    |    |    |    |

    -------------------- Imaginary time evolution

    -------------------- layer n

    |____|____|____|____ mps-representation of all layer-(n-1) circuit

    =

    ______________ new_mps
    |   |   |
    ______________ layer n
    |   |   |
    ______________ mps_old

    '''
    L = len(layer_gate) + 1
    list_of_A_list = apply_U_all([t.copy() for t in mps_old],
                                 layer_gate,
                                 cache=True)
    # we copy the tensor in mps_old, so that mps_old is not modified.
    # [Notice] apply_U_all function modified the A_list inplace.
    # [Todo] maybe change the behavior above. not inplace?
    assert(len(list_of_A_list) == L)
    # L - 1 gate, includding not applying gate
    # idx=0 not applying gate,
    # idx=1, state after applying gate-0 on site-0, site-1.
    # idx=2, state after applying gate-1 on site-1, site-2.
    new_layer = [None] * (L-1)

    for idx in range(L - 2, -1, -1):
        mps_cache = list_of_A_list[idx]
        new_gate = var_gate(new_mps, idx, mps_cache)
        new_layer[idx] = new_gate
        # conjugate the gate
        # <psi|U = (U^\dagger |psi>)^\dagger
        new_gate_conj = new_gate.reshape([4, 4]).T.conj()
        new_gate_conj = new_gate_conj.reshape([2, 2, 2, 2])
        # new_gate_conj = np.einsum('ijkl->klij', new_gate).conj()

        apply_gate(new_mps, new_gate_conj, idx)

    return new_mps, new_layer

def var_gate(new_mps, site, mps_cache):
    '''
    max
    <new_mps | gate | mps_cache>
    where gate is actting on (site, site+1)
    '''
    L = len(new_mps)
    Lp = np.zeros([1, 1])
    Lp[0, 0] = 1.
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, mps_cache[i], axes=(0, 1))
        Lp = np.tensordot(Lp, new_mps[i].conj(), axes=([0, 1], [1,0]))
        Lp_list.append(Lp)

    Rp = np.zeros([1, 1])
    Rp[0, 0] = 1.
    Rp_list = [Rp]

    for i in range(L-1, -1, -1):
        Rp = np.tensordot(mps_cache[i], Rp, axes=(2, 0))
        Rp = np.tensordot(Rp, new_mps[i].conj(), axes=([0, 2], [0, 2]))
        Rp_list.append(Rp)

    L_env = Lp_list[site]
    R_env = Rp_list[L-2-site]

    theta_top = np.tensordot(new_mps[site].conj(), new_mps[site + 1].conj(), axes=(2,1)) # p l, q r
    theta_bot = np.tensordot(mps_cache[site], mps_cache[site + 1],axes=(2,1))

    M = np.tensordot(L_env, theta_bot, axes=([0], [1])) #l, p, q, r
    M = np.tensordot(M, R_env, axes=([3], [0])) #l, p, q, r
    M = np.tensordot(M, theta_top, axes=([0, 3], [1, 3])) #lower_p, lower_q, upper_p, upper_q
    M = M.reshape([4, 4])

    # M_copy = M.reshape([2, 2, 2, 2]).copy()
    # M_copy = M_copy[:, 0, :, :]
    # U, _, Vd = np.linalg.svd(M_copy.reshape([2, 4]), full_matrices=False)
    # new_gate = np.dot(U, Vd).reshape([2, 2, 2])
    # new_gate_ = np.random.rand(2, 2, 2, 2)
    # new_gate_[:, 0, :, :] = new_gate
    # return new_gate_

    # new_gate = polar(M)
    U, _, Vd = np.linalg.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd)
    # new_gate = np.dot(Vd.T.conjugate(), U.T.conjugate())
    new_gate = new_gate.reshape([2, 2, 2, 2])

    return new_gate

def apply_gate(A_list, gate, idx):
    '''
    modification inplace
    '''
    d1,chi1,chi2 = A_list[idx].shape
    d2,chi2,chi3 = A_list[idx + 1].shape

    theta = np.tensordot(A_list[idx], A_list[idx + 1],axes=(2,1))
    theta = np.tensordot(gate, theta, axes=([0,1],[0,2]))
    theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))

    X, Y, Z = np.linalg.svd(theta, full_matrices=0)
    chi2 = np.sum(Y>10.**(-10))

    piv = np.zeros(len(Y), np.bool)
    piv[(np.argsort(Y)[::-1])[:chi2]] = True

    Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
    # Y = Y / invsq
    X = X[:,piv]
    Z = Z[piv,:]

    X=np.reshape(X, (d1, chi1, chi2))
    A_list[idx]   = X.reshape([d1, chi1, chi2])
    A_list[idx + 1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])
    return A_list

def overlap(psi1,psi2):
    N = np.ones([1,1]) # a ap
    L = len(psi1)
    for i in np.arange(L):
        N = np.tensordot(N,np.conj(psi1[i]), axes=(1,1)) # a ap 
        N = np.tensordot(N,psi2[i], axes=([0,1],[1,0])) # ap a
        N = N.transpose(1,0)
    N = np.trace(N)
    return(N)

def expectation_values(A_list, H_list, check_norm=True):
    if check_norm:
        assert np.isclose(overlap(A_list, A_list), 1.)
    else:
        pass

    L = len(A_list)
    Lp = np.zeros([1, 1])
    Lp[0, 0] = 1.
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, A_list[i], axes=(0, 1)) # ap i b
        Lp = np.tensordot(Lp, A_list[i].conj(), axes=([0, 1], [1,0])) # b bp
        Lp_list.append(Lp)

    Rp = np.zeros([1, 1])
    Rp[0, 0] = 1.

    E_list = []
    for i in range(L - 2, -1, -1):
        Rp = np.tensordot(A_list[i+1].conj(),Rp, axes=(2, 1)) #[p,l,r] [d,u] -> [p,l,d]
        E = np.tensordot(A_list[i].conj(),Rp, axes=(2, 1)) #[p,l,r] [q,L,R] -> [p,l,q,R]
        E = np.tensordot(H_list[i],E, axes=([0,1], [0,2])) # [p,q, r,s] , [p,l,q,R] -> [r,s, l,R]
        E = np.tensordot(A_list[i+1],E, axes=([0,2], [1,3])) # [s, L, R], [r,s, l,R] -> [L, r, l]
        E = np.tensordot(A_list[i],E, axes=([0,2], [1,0])) # [r, ll, L] [L, r, l] -> [ll, l]
        E = np.tensordot(Lp_list[i],E, axes=([0,1],[0,1]))
        Rp = np.tensordot(A_list[i+1],Rp,axes=([0,2], [0,2]))
        E_list.append(E.item())

    return E_list

def apply_U_all(A_list, U_list, cache=False):
    '''
    if cache is True, we will return a list_A_list,
    which gives the list of mps of length L, which corresponds to
    applying 0, 1, 2, ... L-1 gates.

    '''
    L = len(A_list)
    if cache:
        list_A_list = []
        list_A_list.append([A.copy() for A in A_list])

    for i in range(L-1):
        d1,chi1,chi2 = A_list[i].shape
        d2,chi2,chi3 = A_list[i+1].shape

        theta = np.tensordot(A_list[i],A_list[i+1],axes=(2,1))
        theta = np.tensordot(U_list[i],theta,axes=([0,1],[0,2]))
        theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))

        X, Y, Z = np.linalg.svd(theta, full_matrices=0)
        chi2 = np.sum(Y>10.**(-10))

        piv = np.zeros(len(Y), np.bool)
        piv[(np.argsort(Y)[::-1])[:chi2]] = True

        Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        # Y = Y / invsq
        X = X[:,piv]
        Z = Z[piv,:]

        X=np.reshape(X, (d1, chi1, chi2))
        A_list[i]   = X.reshape([d1, chi1, chi2])
        A_list[i+1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])
        if cache:
            list_A_list.append([A.copy() for A in A_list])
        else:
            pass

    if cache:
        return list_A_list
    else:
        return A_list

def apply_U(A_list, U_list, onset):
    '''
    There are two subset of gate.
    onset indicate whether we are applying even (0, 2, 4, ...)
    or odd (1, 3, 5, ...) gates
    '''
    L = len(A_list)

    Ap_list = [None for i in range(L)]
    if onset == 1:
        Ap_list[0] = A_list[0]
        Ap_list[L-1] = A_list[L-1]

    for i in range(onset,L-1,2):
        d1,chi1,chi2 = A_list[i].shape
        d2,chi2,chi3 = A_list[i+1].shape

        theta = np.tensordot(A_list[i],A_list[i+1],axes=(2,1))
        theta = np.tensordot(U_list[i],theta,axes=([0,1],[0,2]))
        theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))

        X, Y, Z = np.linalg.svd(theta,full_matrices=0)
        chi2 = np.sum(Y>10.**(-10))

        piv = np.zeros(len(Y), np.bool)
        piv[(np.argsort(Y)[::-1])[:chi2]] = True

        Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        X = X[:,piv]
        Z = Z[piv,:]

        X=np.reshape(X, (d1, chi1, chi2))
        Ap_list[i]   = X.reshape([d1, chi1, chi2])
        Ap_list[i+1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])

    return Ap_list

def var_A(A_list, Ap_list):
    L = len(A_list)
    Lp = np.zeros([1, 1])
    Lp[0, 0] = 1.
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, A_list[i], axes=(0, 1))
        Lp = np.tensordot(Lp, Ap_list[i].conj(), axes=([0, 1], [1,0]))
        Lp_list.append(Lp)

    Rp = np.zeros([1, 1])
    Rp[0, 0] = 1.

    A_list_new = [[] for i in range(L)]
    for i in range(L - 1, -1, -1):
        Rp = np.tensordot(Ap_list[i].conj(),Rp, axes=(2, 1))
        theta = np.tensordot(Lp_list[i],Rp, axes=(1,1))
        theta = theta.transpose(1,0,2)
        A_list_new[i] = polar(theta)
        Rp = np.tensordot(A_list_new[i], Rp, axes=([0,2], [0,2]))

    return A_list_new

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
    H_list  =  get_H_TFI(L, J, g)
    # E_exact =  ed.get_E_Ising_exact(g,J,L)
    N_iter = int(sys.argv[4])
    order = str(sys.argv[5])

    # second_order = True
    # if second_order:
    #     order = '2nd'
    # else:
    #     order = '1st'

    my_circuit = []

    # delta_list = [np.sum(expectation_values(A_list, H_list))-E_exact.item()]
    t_list = [0]
    E_list = []
    update_error_list = [0.]

    for dep_idx in range(depth):
        # if dep_idx > 0:
        #     identity_layer = [np.eye(4).reshape([2, 2, 2, 2]) for i in range(L-1)]
        #     my_circuit.append(identity_layer)
        # else:
        #     random_layer = [random_2site_U(2) for i in range(L-1)]
        #     my_circuit.append(random_layer)
        random_layer = [random_2site_U(2) for i in range(L-1)]
        my_circuit.append(random_layer)
        current_depth = dep_idx + 1

    mps_of_layer = circuit_2_mps(my_circuit)
    E_list.append(np.sum(expectation_values(mps_of_layer[-1], H_list)))

    for dt in [0.05,0.01,0.001]:
        U_list =  make_U(H_list, dt)
        U_half_list =  make_U(H_list, dt/2.)
        for i in range(int(20//dt**(0.75))):
            mps_of_layer = circuit_2_mps(my_circuit)
            mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
            # [TODO] remove the assertion below
            assert np.isclose(overlap(mps_of_last_layer, mps_of_last_layer), 1.)
            if order == '2nd':
                new_mps = apply_U(mps_of_last_layer,  U_half_list, 0)
                new_mps = apply_U(new_mps, U_list, 1)
                new_mps = apply_U(new_mps, U_half_list, 0)
            else:
                new_mps = apply_U(mps_of_last_layer,  U_list, 0)
                new_mps = apply_U(new_mps, U_list, 1)

            print("Norm new mps = ", overlap(new_mps, new_mps), "new state aimed E = ",
                  np.sum(expectation_values(new_mps, H_list, check_norm=False))/overlap(new_mps, new_mps)
                 )
            # new_mps is the e(-H)|psi0> which is not normalizaed.

            for iter_idx in range(N_iter):
                iter_mps = [A.copy() for A in new_mps]
                for var_dep_idx in range(current_depth, 0, -1):
                # for var_dep_idx in range(current_depth, current_depth-1, -1):
                    # circuit is modified inplace
                    # new mps is returned
                    iter_mps, new_layer = var_layer([A.copy() for A in iter_mps],
                                                    my_circuit[var_dep_idx - 1],
                                                    mps_of_layer[var_dep_idx - 1],
                                                   )
                    assert(len(new_layer) == L -1)
                    my_circuit[var_dep_idx - 1] = new_layer

                mps_of_layer = circuit_2_mps(my_circuit)

            # [Todo] log the fedility here
            mps_of_layer = circuit_2_mps(my_circuit)
            mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]
            assert np.isclose(overlap(mps_of_last_layer, mps_of_last_layer), 1.)
            current_energy = np.sum(expectation_values(mps_of_last_layer, H_list))
            # delta_list.append(np.sum(expectation_values(mps_of_last_layer, H_list))-E_exact.item())
            E_list.append(current_energy)
            t_list.append(t_list[-1]+dt)
            # print(t_list[-1],delta_list[-1])

            fidelity_reached = np.abs(overlap(new_mps, mps_of_last_layer))**2 / overlap(new_mps, new_mps)
            print("fidelity reached : ", fidelity_reached)
            update_error_list.append(1. - fidelity_reached)



    dir_path = 'data/1d_TFI_g%.1f/L%d/' % (g, L)
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

    dir_path = 'data/1d_TFI_g%.1f/' % (g)
    best_E = np.amin(E_list)
    filename = 'circuit_depth%d_Niter%d_%s_energy.csv' % (depth, N_iter, order)
    path = dir_path + filename
    # Try to load file 
    # If data return
    E_dict = {}
    try:
        E_array = misc.load_array(path)
        E_dict = misc.nparray_2_dict(E_array)
        assert L in E_dict.keys()
        print("Found data")
    except Exception as error:
        print(error)
        E_dict[L] = best_E
        misc.save_array(path, misc.dict_2_nparray(E_dict))
        # If no data --> generate data
        print("Save new data")


