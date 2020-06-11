from scipy import integrate
from scipy.linalg import expm
import misc, os
import sys
## 
## We use jax.numpy if possible
## or autograd.numpy
##
## Regarding wether choosing to use autograd or jax
## see https://github.com/google/jax/issues/193
## Basically, if we do not use gpu or tpu and work with 
## small problem size and do not want to spend time fixing
## function with @jit, then autograd should be fine.

try:
    raise
    # import jax.numpy as np
    import autograd.numpy as np
    from autograd import grad
    # from jax import random
    # from jax import grad, jit, vmap
    # from jax.config import config
    # config.update("jax_enable_x64", True)
except:
    import numpy as np
    np.seterr(all='raise')
    print("some function may be broken")

import numpy as onp


def get_H(Hamiltonian, L, J, g, h=0):
    if Hamiltonian == 'TFI':
        return get_H_TFI(L, J, g, h)
    elif Hamiltonian == 'XXZ':
        return get_H_XXZ(L, J, g)
    else:
        raise

def get_H_TFI(L, J, g, h=0):
    '''
    H_TFI = - J ZZ - g X
    H_TFI = - J XX - g Z - h X
    Return:
        H: list of local hamiltonian
    '''
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    eye2 = np.eye(2)
    d = 2

    def hamiltonian(gl, gr, hl, hr, J):
        # return (-np.kron(sz, sz) * J - gr * np.kron(eye2, sx) - gl * np.kron(sx, eye2)).reshape([d] * 4)
        return (-np.kron(sx, sx) * J - gr * np.kron(eye2, sz) - gl * np.kron(sz, eye2)
                -hr * np.kron(eye2, sx) - hl * np.kron(sx, eye2)
               ).reshape([d] * 4)

    H = []
    for j in range(L - 1):
        if j == 0:
            gl = g
            hl = h
        else:
            gl = 0.5 * g
            hl = 0.5 * h
        if j == L - 2:
            gr = 1. * g
            hr = 1. * h
        else:
            gr = 0.5 * g
            hr = 0.5 * h

        H.append(hamiltonian(gl, gr, hl, hr, J))
    return H

def get_H_XXZ(L, J, g):
    '''
    H_XXZX = J XX + J YY + Jg ZZ
    '''
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1.j], [1.j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    eye2 = np.eye(2)
    d = 2

    def hamiltonian(g, J):
        return (np.kron(sx, sx) * J + np.kron(sy, sy) * J + np.kron(sz, sz) * J * g ).reshape([d] * 4)

    H = []
    for j in range(L - 1):
        H.append(hamiltonian(g, J).real)

    return H

def make_U(H, t):
    """ U = exp(-t H) """
    d = H[0].shape[0]
    return [expm(-t * h.reshape((d**2, -1))).reshape([d] * 4) for h in H]

def init_mps(L, chi, d):
    '''
    Return MPS in AAAAAA form, i.e. left canonical form
    such that \sum A^\dagger A = I
    '''
    A_list = []
    for i in range(L):
        chi1 = np.min([d**np.min([i,L-i]),chi])
        chi2 = np.min([d**np.min([i+1,L-i-1]),chi])
        try:
            A_list.append(polar(0.5 - np.random.uniform(size=[d,chi1,chi2])))
            # A_list.append(polar(np.random.uniform(size=[d,chi1,chi2])))
        except:
            A_list.append(polar(0.5 - onp.random.uniform(size=[d,chi1,chi2])))
            # A_list.append(polar(onp.random.uniform(size=[d,chi1,chi2])))

    return A_list

def polar(A):
    '''
    return:
        polar decomposition of A
    '''
    d,chi1,chi2 = A.shape
    Y,D,Z = misc.svd(A.reshape(d*chi1,chi2), full_matrices=False)
    return np.dot(Y,Z).reshape([d,chi1,chi2])

def random_2site_U(d, factor=1e-2):
    try:
        A = np.random.uniform(size=[d**2, d**2]) * factor
    except:
        A = onp.random.uniform(size=[d**2, d**2]) * factor

    A = A-A.T
    U = (np.eye(d**2)-A).dot(np.linalg.inv(np.eye(d**2)+A))
    return U.reshape([d] * 4)
    # M = onp.random.rand(d ** 2, d ** 2)
    # Q, _ = np.linalg.qr(0.5 - M)
    # return Q.reshape([d] * 4)

def circuit_2_mps(circuit, product_state, chi=None):
    #[TODO] extend this to brickwall
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
        list of mps representation of each layer; each in left canonical form
        mps_of_layer[0] gives the product state |psi0>
        mps_of_layer[1] gives the U(0) |psi0>
    '''
    depth = len(circuit)
    L = len(circuit[0]) + 1
    # A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    A_list = [t.copy() for t in product_state]

    mps_of_layer = []
    mps_of_layer.append([A.copy() for A in A_list])
    # A_list is modified inplace always.
    # so we always have to copy A tensors.
    for dep_idx in range(depth):
        U_list = circuit[dep_idx]
        ### A_list is modified inplace
        right_canonicalize(A_list)
        A_list, trunc_error = apply_U_all(A_list, U_list, cache=False)
        mps_of_layer.append([A.copy() for A in A_list])

    return mps_of_layer

def var_circuit(target_mps, bottom_mps, circuit, product_state):
    '''
    Goal:
        Do a sweep from top of the circuit down to product state,
        and do a sweep from bottom of the circuit to top.

        When sweeping from top to bottom, we take the target and bottom mps,
        we remove one unitary at a time by applying the conjugate unitary.
        We then form the environment. Once a new unitary is obtained, it is
        applied to the "bra", i.e. top mps.

        When sweeping from bottom to top, we take the product state, and the
        top mps resulting from the first part of the algorithm.
        We remove one unitary at a time from the top mps by applying the
        conjugate unitary. We then form the environment. Once a new unitary
        is obtained, it is applied to the bottom mps.

        In this approach, we do not need to store the intermediate mps representation.
        We avoid it by the removing unitary approach.
        This however, has a disadvantage that the unitary is not cancelled completely.
        For example the bottom mps after sweeping from top to bottom usually is not a
        product state, but bond dimension 2.
        The top mps would have bond dimension that is larger than necessary.

        The other approach using caching mps and avoid unitary removal is implemented
        in var_circuit_2 function.

    Input:
        target_mps: can be not normalized, but should be in left canonical form.
        bottom_mps: the mps representing the contraction of full circuit
            with product state
        circuit: list of list of unitary
        product_state: the initial product state that the circuit acts on.
    Output:
        mps_final: the mps representation of the updated circuit
        circuit: list of list of unitary
        product_state: the initial product state
    '''
    current_depth = len(circuit)
    L = len(circuit[0]) + 1
    top_mps = [A.copy() for A in target_mps]

    print("Sweeping from top to bottom, overlap (before) : ",
          overlap(target_mps, bottom_mps))
    for var_dep_idx in range(current_depth-1, -1, -1):
        Lp_cache = [np.ones([1, 1])] + [None] * (L-1)
        Rp_cache = [None] * (L-1) + [np.ones([1, 1])]
        for idx in range(L - 2, -1, -1):
            remove_gate = circuit[var_dep_idx][idx]
            remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
            remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])
            apply_gate(bottom_mps, remove_gate_conj, idx, move='left')
            # now bottom_mps is mps without remove_gate,
            # we can now variational finding the optimal gate to replace it.

            new_gate, Lp_cache, Rp_cache = var_gate_w_cache(top_mps, idx, bottom_mps, Lp_cache, Rp_cache)
            circuit[var_dep_idx][idx] = new_gate

            # conjugate the gate
            # <psi|U = (U^\dagger |psi>)^\dagger
            new_gate_conj = new_gate.reshape([4, 4]).T.conj()
            new_gate_conj = new_gate_conj.reshape([2, 2, 2, 2])
            # new_gate_conj = np.einsum('ijkl->klij', new_gate).conj()

            apply_gate(top_mps, new_gate_conj, idx, move='left')


        left_canonicalize(top_mps)
        left_canonicalize(bottom_mps)

    max_chi_bot = np.amax([np.amax(t.shape) for t in bottom_mps])
    max_chi_top = np.amax([np.amax(t.shape) for t in top_mps])
    print("after sweep down, X(top_mps) = ", max_chi_top, " X(bot_mps) = ", max_chi_bot)


    overlap_abs = np.abs(overlap(bottom_mps, product_state))
    # print("overlap_abs = ", overlap_abs)
    assert np.isclose(overlap_abs, 1, rtol=1e-8)
    bottom_mps = [t.copy() for t in product_state]

    print("Sweeping from bottom to top, overlap (before) : ",
          overlap(top_mps, bottom_mps)
         )
    for var_dep_idx in range(0, current_depth):
        right_canonicalize(top_mps)
        right_canonicalize(bottom_mps)

        Lp_cache = [np.ones([1, 1])] + [None] * (L-1)
        Rp_cache = [None] * (L-1) + [np.ones([1, 1])]
        for idx in range(L-1):
            gate = circuit[var_dep_idx][idx]
            apply_gate(top_mps, gate, idx)
            ## This remove the gate from top_mps
            ## Because <\phi | U_{ij} | \psi> = inner( U_{ij}^\dagger |\phi>, |\psi> ) 
            ## applying U would remove the U_{ij}^\dagger, and
            ## partial_Tr[ |\phi>, |\psi> ] = Env

            # for t_list in [top_mps, bottom_mps, Lp_cache, Rp_cache]:
            #     for t in t_list:
            #         if t is not None:
            #             if not np.isfinite(t).all():
            #                 print("check before")
            #                 import pdb;pdb.set_trace()

            new_gate, Lp_cache, Rp_cache = var_gate_w_cache(top_mps, idx, bottom_mps, Lp_cache, Rp_cache)
            # if not np.isfinite(new_gate).all():
            #     print("new gate wrong")
            #     import pdb;pdb.set_trace()

            # for t_list in [top_mps, bottom_mps, Lp_cache, Rp_cache]:
            #     for t in t_list:
            #         if t is not None:
            #             if not np.isfinite(t).all():
            #                 print("check after")
            #                 import pdb;pdb.set_trace()

            circuit[var_dep_idx][idx] = new_gate

            apply_gate(bottom_mps, new_gate, idx)

    ## finish sweeping
    ## bottom_mps is mps_final

    max_chi_bot = np.amax([np.amax(t.shape) for t in bottom_mps])
    max_chi_top = np.amax([np.amax(t.shape) for t in top_mps])
    print("after sweep up, X(top_mps) = ", max_chi_top, " X(bot_mps) = ", max_chi_bot)
    return bottom_mps, circuit

def var_circuit2(target_mps, product_state, circuit):
    """
    Goal:
        Do a sweep updating gates from top of the circuit down to product state,
        and do a sweep updating gates from bottom of the circuit to top.

    Input:
        target_mps: can be not normalized, but should be in left canonical form.
        product_state
        circuit

    Output:
        new_circuit
    """
    L = len(target_mps)
    circuit_depth = len(circuit)

    top_mps = [t.copy() for t in target_mps]  # in left canonical form
    bottom_mps_cache = circuit_2_mps(circuit, product_state) # in left canonical form
    # All mps in bottom_mps_cache is in left canonical form

    #[TODO] Is this necessary?
    # When append to top_mps_cache, we always need to use copy
    # such that the cache is really an independent list.
    # (The other way may be to create a new list everytime in var_layer and return it.)
    top_mps_cache = []
    top_mps_cache.append([t.copy() for t in top_mps])

    for dep_idx in range(circuit_depth-1, -1, -1):
        top_mps, new_layer = var_layer(top_mps,
                                       circuit[dep_idx],
                                       bottom_mps_cache[dep_idx],
                                       direction='down'
                                      )
        assert(len(new_layer) == L-1)
        circuit[dep_idx] = new_layer
        top_mps_cache.append([t.copy() for t in top_mps])

    assert( len(top_mps_cache)  == circuit_depth + 1)
    # top_mps_cache [0] --> target_mps
    # top_mps_cache [1] --> target_mps + 1-layer
    # top_mps_cache [x] --> target_mps + x-layer
    # [TODO]
    bottom_mps = bottom_mps_cache[0]
    for dep_idx in range(0, circuit_depth):
        bottom_mps, new_layer = var_layer(top_mps_cache[-2-dep_idx],
                                          circuit[dep_idx],
                                          bottom_mps,
                                          direction='up'
                                         )
        circuit[dep_idx] = new_layer

    return bottom_mps, circuit

def var_layer(top_mps, layer_gate, bottom_mps, direction, list_of_A_list=None):
    '''
    Goal:
        See graphical representation below
    Input:
        top_mps: in left canonical form
        layer_gate:
        bottom_mps: in left canonical form
        list_of_A_list
    Output:
        top_mps, new_layer


    We try to maximize the quantity:

    <top_mps | layer_gate | bottom_mps>

    =

    |    |    |    |
    -------------------- layer 0

    -------------------- layer 1
    .
    .
    -------------------- layer n

    -------------------- Imaginary time evolution

    -------------------- layer n
    .
    .
    -------------------- layer x
    .
    .
    -------------------- layer 1

    -------------------- layer 0
    |    |    |    |

    =

    -------------------- mps-representation of all layer-(n) circuit
    |    |    |    |

    -------------------- Imaginary time evolution
    .
    .
    -------------------- layer x+1

    -------------------- layer x

    |____|____|____|____ mps-representation of all layer-(x-1) circuit

    =

    ______________ top_mps
    |   |   |
    ______________ layer x
    |   |   |
    ______________ bottom_mps

    '''
    L = len(layer_gate) + 1
    if direction == 'down':
        # Form upward cache
        if list_of_A_list is None:
            # we copy the tensor in bottom_mps, so that bottom_mps is not modified.
            # [Notice] apply_U_all function modified the A_list inplace.
            # [Todo] maybe change the behavior above. not inplace?
            A_list, trunc_err = right_canonicalize([t.copy() for t in bottom_mps])
            list_of_A_list, trunc_error = apply_U_all(A_list,
                                                      layer_gate,
                                                      cache=True)
        else:
            pass

        assert(len(list_of_A_list) == L)
        # There are L states, because with L-1 gates, including not applying gate
        # idx=0 not applying gate,
        # idx=1, state after applying gate-0 on site-0, site-1.
        # idx=2, state after applying gate-1 on site-1, site-2.
        new_layer = [None] * (L-1)

        for idx in range(L - 2, -1, -1):
            mps_cache = list_of_A_list[idx]
            # [TODO] To add brickwall condition: if brickwall, some return directly identity
            new_gate = var_gate(top_mps, idx, mps_cache)
            new_layer[idx] = new_gate
            # conjugate the gate
            # <psi|U = (U^\dagger |psi>)^\dagger
            new_gate_conj = new_gate.reshape([4, 4]).T.conj()
            new_gate_conj = new_gate_conj.reshape([2, 2, 2, 2])
            # new_gate_conj = np.einsum('ijkl->klij', new_gate).conj()

            apply_gate(top_mps, new_gate_conj, idx, move='left')

        # top_mps ends up in right canonical form
        top_mps, trunc_err = left_canonicalize(top_mps)
        # top_mps brought back to left canonical form
        assert trunc_err < 1e-12

        return top_mps, new_layer
    elif direction == 'up':
        # Form downward cache
        downward_cache_list = []
        downward_cache_list.append([t.copy() for t in top_mps])
        for idx in range(L-2, -1, -1):
            conj_gate = layer_gate[idx].reshape([4,4]).T.conj()
            conj_gate = conj_gate.reshape([2, 2, 2, 2])
            apply_gate(top_mps, conj_gate, idx, move='left')
            downward_cache_list.append([t.copy() for t in top_mps])

        assert(len(downward_cache_list) == L)
        new_layer = [None] * (L-1)

        bottom_mps, trunc_err = right_canonicalize([t.copy() for t in bottom_mps])
        for idx in range(L-1):
            top_mps = downward_cache_list[-2-idx]
            new_gate = var_gate(top_mps, idx, bottom_mps)
            new_layer[idx] = new_gate
            apply_gate(bottom_mps, new_gate, idx, move='right')

        return bottom_mps, new_layer

    else:
        raise NotImplementedError

def var_gate_w_cache(new_mps, site, mps_ket, Lp_cache, Rp_cache):
    '''
    max
    <new_mps | gate | mps_ket>
    where gate is actting on (site, site+1)
    '''
    L = len(new_mps)
    # Lp = np.ones([1, 1])
    # Lp_list = [Lp]
    # Lp_cache = [Lp, ... ]

    for i in range(site):
        Lp = Lp_cache[i]
        if Lp_cache[i+1] is None:
            Lp = np.tensordot(Lp, mps_ket[i], axes=(0, 1))
            Lp = np.tensordot(Lp, new_mps[i].conj(), axes=([0, 1], [1,0]))
            Lp_cache[i+1] = Lp
        else:
            pass

    # Rp = np.ones([1, 1])
    # Rp_list = [Rp]

    for i in range(L-1, site+1, -1):
        Rp = Rp_cache[i]
        if Rp_cache[i-1] is None:
            Rp = np.tensordot(mps_ket[i], Rp, axes=(2, 0))
            Rp = np.tensordot(Rp, new_mps[i].conj(), axes=([0, 2], [0, 2]))
            Rp_cache[i-1] = Rp

    L_env = Lp_cache[site]
    # L_env = Lp_list[site]
    R_env = Rp_cache[site+1]
    # R_env = Rp_list[L-2-site]

    theta_top = np.tensordot(new_mps[site].conj(), new_mps[site + 1].conj(), axes=(2,1)) # p l, q r
    theta_bot = np.tensordot(mps_ket[site], mps_ket[site + 1],axes=(2,1))

    M = np.tensordot(L_env, theta_bot, axes=([0], [1])) #l, p, q, r
    M = np.tensordot(M, R_env, axes=([3], [0])) #l, p, q, r
    M = np.tensordot(M, theta_top, axes=([0, 3], [1, 3])) #lower_p, lower_q, upper_p, upper_q
    M = M.reshape([4, 4])

    ### For detailed explanation of the formula, see function var_gate
    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])

    return new_gate, Lp_cache, Rp_cache

def var_gate(new_mps, site, mps_ket):
    '''
    Input:
        new_mps : the mps representation of the bra
        site : two site gate is applied on (site, site+1)
        mps_ket : the mps representation of the ket
    Goal:
        max  Re<new_mps | gate | mps_ket>
        where gate is actting on (site, site+1)
    return:
        new_gate
    '''
    L = len(new_mps)
    Lp = np.ones([1, 1])
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, mps_ket[i], axes=(0, 1))
        Lp = np.tensordot(Lp, new_mps[i].conj(), axes=([0, 1], [1,0]))
        Lp_list.append(Lp)

    Rp = np.ones([1, 1])
    Rp_list = [Rp]

    for i in range(L-1, -1, -1):
        Rp = np.tensordot(mps_ket[i], Rp, axes=(2, 0))
        Rp = np.tensordot(Rp, new_mps[i].conj(), axes=([0, 2], [0, 2]))
        Rp_list.append(Rp)

    L_env = Lp_list[site]
    R_env = Rp_list[L-2-site]

    theta_top = np.tensordot(new_mps[site].conj(), new_mps[site + 1].conj(), axes=(2,1)) # p l, q r
    theta_bot = np.tensordot(mps_ket[site], mps_ket[site + 1],axes=(2,1))

    M = np.tensordot(L_env, theta_bot, axes=([0], [1])) #l, p, q, r
    M = np.tensordot(M, R_env, axes=([3], [0])) #l, p, q, r
    M = np.tensordot(M, theta_top, axes=([0, 3], [1, 3])) #lower_p, lower_q, upper_p, upper_q
    M = M.reshape([4, 4])

    ##### The formula should work for the first layer; where the unitary there has redundant
    ##### degree of freedom.
    # M_copy = M.reshape([2, 2, 2, 2]).copy()
    # M_copy = M_copy[:, 0, :, :]
    # U, _, Vd = misc.svd(M_copy.reshape([2, 4]), full_matrices=False)
    # new_gate = np.dot(U, Vd).reshape([2, 2, 2])
    # new_gate_ = onp.random.rand(2, 2, 2, 2) * (1+0j)
    # new_gate_[:, 0, :, :] = new_gate.conj()
    # # return new_gate_

    ######################################################################
    # new_gate = polar(M)
    # We are maximizing Re[\sum_ij A_ij W_ij ] with W^\dagger W = I
    # Re[\sum_ij A_ij W_ij] = Re Tr[ WA^T], A=USV^dagger, A^T = V*SU^T
    # W = (UV^\dagger)* = U* V^T   gives optimal results.
    # Re Tr[ WA^T] = Tr[S]
    ######################################################################
    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])

    return new_gate

def apply_gate(A_list, gate, idx, move):
    '''
    modification inplace
    '''
    d1,chi1,chi2 = A_list[idx].shape
    d2,chi2,chi3 = A_list[idx + 1].shape

    theta = np.tensordot(A_list[idx], A_list[idx + 1],axes=(2,1))
    theta = np.tensordot(gate, theta, axes=([0,1],[0,2]))
    theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))

    X, Y, Z = misc.svd(theta, full_matrices=0)
    chi2 = np.sum(Y>1e-14)

    # piv = np.zeros(len(Y), onp.bool)
    # piv[(np.argsort(Y)[::-1])[:chi2]] = True

    # Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
    # # Y = Y / invsq
    # X = X[:,piv]
    # Z = Z[piv,:]

    arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
    Y = Y[arg_sorted_idx]  # chi2
    X = X[: ,arg_sorted_idx]  # (d1*chi1, chi2)
    Z = Z[arg_sorted_idx, :]  # (chi2, d2*chi3)

    if move == 'right':
        X=np.reshape(X, (d1, chi1, chi2))
        A_list[idx]   = X.reshape([d1, chi1, chi2])
        A_list[idx + 1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])
    elif move == 'left':
        A_list[idx + 1]   = np.transpose(Z.reshape([chi2, d2, chi3]), [1, 0, 2])
        A_list[idx] = np.dot(X, np.diag(Y)).reshape([d1, chi1, chi2])
    else:
        raise

    # for t in A_list:
    #     try:
    #         assert np.isfinite(t).all()
    #     except:
    #         import pdb;pdb.set_trace()

    return A_list

def overlap(psi1,psi2):
    N = np.ones([1,1]) # a ap
    L = len(psi1)
    for i in np.arange(L):
        N = np.tensordot(N,np.conj(psi1[i]), axes=(1,1)) # a ap 
        N = np.tensordot(N,psi2[i], axes=([0,1],[1,0])) # ap a
        N = np.transpose(N, [1,0])
    N = np.trace(N)
    return(N)

def expectation_values_1_site(A_list, Op_list, check_norm=True):
    if check_norm:
        assert np.isclose(overlap(A_list, A_list), 1.)
    else:
        pass

    L = len(A_list)
    Lp = np.ones([1, 1])
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, A_list[i], axes=(0, 1)) # ap i b
        Lp = np.tensordot(Lp, np.conj(A_list[i]), axes=([0, 1], [1,0])) # b bp
        Lp_list.append(Lp)

    Rp = np.ones([1, 1])

    Op_per_site = np.zeros([L], dtype=np.complex)
    for i in range(L - 2, -2, -1):
        Rp = np.tensordot(np.conj(A_list[i+1]), Rp, axes=(2, 1)) #[p,l,r] [d,u] -> [p,l,d]
        Op = np.tensordot(Op_list[i+1], Rp, axes=(1, 0)) #[p,q], [q,l,d] -> [p,l,d]
        Op = np.tensordot(Op, A_list[i+1], axes=([0,2], [0,2])) #[p,ul,d], [p,dl,rr] -> [ul, dl]
        Op = np.tensordot(Lp_list[i+1], Op, axes=([0,1], [1,0]))
        Op_per_site[i+1] = Op[None][0]

        Rp = np.tensordot(A_list[i+1],Rp,axes=([0,2], [0,2]))

    return Op_per_site

def expectation_values(A_list, H_list, check_norm=True):
    if check_norm:
        assert np.isclose(np.abs(overlap(A_list, A_list)), 1.)
    else:
        pass

    L = len(A_list)
    Lp = np.ones([1, 1])
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, A_list[i], axes=(0, 1)) # ap i b
        Lp = np.tensordot(Lp, np.conj(A_list[i]), axes=([0, 1], [1,0])) # b bp
        Lp_list.append(Lp)

    Rp = np.ones([1, 1])

    E_list = []
    for i in range(L - 2, -1, -1):
        Rp = np.tensordot(np.conj(A_list[i+1]), Rp, axes=(2, 1)) #[p,l,r] [d,u] -> [p,l,d]
        E = np.tensordot(np.conj(A_list[i]), Rp, axes=(2, 1)) #[p,l,r] [q,L,R] -> [p,l,q,R]
        E = np.tensordot(H_list[i],E, axes=([0,1], [0,2])) # [p,q, r,s] , [p,l,q,R] -> [r,s, l,R]
        E = np.tensordot(A_list[i+1],E, axes=([0,2], [1,3])) # [s, L, R], [r,s, l,R] -> [L, r, l]
        E = np.tensordot(A_list[i],E, axes=([0,2], [1,0])) # [r, ll, L] [L, r, l] -> [ll, l]
        E = np.tensordot(Lp_list[i],E, axes=([0,1],[0,1]))
        Rp = np.tensordot(A_list[i+1],Rp,axes=([0,2], [0,2]))
        E_list.append(E[None][0])

    return E_list

def right_canonicalize(A_list, no_trunc=False, chi=None):
    '''
    Bring mps in right canonical form, assuming the input mps is in
    left canonical form already.

    modification in place
    '''
    L = len(A_list)
    tot_trunc_err = 0.
    for i in range(L-1, 0, -1):
        d1, chi1, chi2 = A_list[i].shape
        X, Y, Z = misc.svd(np.reshape(np.transpose(A_list[i], [1, 0, 2]), [chi1, d1 * chi2]),
                                full_matrices=0)

        if no_trunc:
            chi1 = np.size(Y)
        else:
            chi1 = np.sum(Y>1e-14)

        if chi is not None:
            chi1 = np.amin([chi1, chi])

        trunc_idx = (np.argsort(Y)[::-1])[chi1:]
        trunc_error = np.sum(Y[trunc_idx] ** 2) / np.sum(Y ** 2)
        tot_trunc_err = tot_trunc_err + trunc_error

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi1]
        Y = Y[arg_sorted_idx]
        Y = Y / np.linalg.norm(Y)
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        A_list[i]   = np.transpose(Z.reshape([chi1, d1, chi2]), [1, 0, 2])
        R = np.dot(X, np.diag(Y))
        new_A = np.tensordot(A_list[i-1], R, axes=([2], [0]))  #[p, 1l, (1r)] [(2l), 2r]
        A_list[i-1] = new_A

    A_list[0] = A_list[0] / np.linalg.norm(A_list[0])
    return A_list, tot_trunc_err

def left_canonicalize(A_list, no_trunc=False, chi=None):
    '''
    Bring mps in left canonical form, assuming the input mps is in
    right canonical form already.

    modification in place
    '''
    L = len(A_list)
    tot_trunc_err = 0
    for i in range(L-1):
        d1, chi1, chi2 = A_list[i].shape
        X, Y, Z = misc.svd(np.reshape(A_list[i], [d1 * chi1, chi2]),
                                full_matrices=0)

        if no_trunc:
            chi2 = np.size(Y)
        else:
            chi2 = np.sum(Y>1e-14)

        if chi is not None:
            chi2 = np.amin([chi2, chi])

        trunc_idx = (np.argsort(Y)[::-1])[chi2:]
        trunc_error = np.sum(Y[trunc_idx] ** 2) / np.sum(Y ** 2)
        tot_trunc_err = tot_trunc_err + trunc_error

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
        Y = Y[arg_sorted_idx]
        Y = Y / np.linalg.norm(Y)
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        A_list[i]   = X.reshape([d1, chi1, chi2])
        R = np.dot(np.diag(Y), Z)
        new_A = np.tensordot(R, A_list[i+1], axes=([1], [1]))  #[1l,(1r)],[p, (2l), 2r]
        A_list[i+1] = np.transpose(new_A, [1, 0, 2])

    A_list[-1] = A_list[-1] / np.linalg.norm(A_list[-1])
    return A_list, tot_trunc_err

def get_entanglement(A_list):
    '''
    Goal:
        Compute the bibpartite entanglement at each cut.
    Input:
        mps in left canonical form
    Output:
        list of bipartite entanglement [(0,1...), (01,2...), (012,...)]
    '''
    L = len(A_list)
    copy_A_list = [A.copy() for A in A_list]
    ent_list = [None] * (L-1)
    for i in range(L-1, 0, -1):
        d1, chi1, chi2 = copy_A_list[i].shape
        X, Y, Z = misc.svd(np.reshape(np.transpose(copy_A_list[i], [1, 0, 2]), [chi1, d1 * chi2]),
                                full_matrices=0)

        chi1 = np.sum(Y>1e-14)

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi1]
        Y = Y[arg_sorted_idx]
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        copy_A_list[i]   = np.transpose(Z.reshape([chi1, d1, chi2]), [1, 0, 2])
        R = np.dot(X, np.diag(Y))
        new_A = np.tensordot(copy_A_list[i-1], R, axes=([2], [0]))  #[p, 1l, (1r)] [(2l), 2r]
        copy_A_list[i-1] = new_A

        bi_ent = -(Y**2).dot(np.log(Y**2))
        ent_list[i-1] = bi_ent

    return ent_list

def apply_U_all(A_list, U_list, cache=False, no_trunc=False, chi=None):
    #[TODO] Rewrite the function by using apply_gate. Make apply_gate more general
    # with trunc, chi argument.
    # Make this function as an application of apply_gate only.
    #[TODO] Write also applying gates backward (L-2, L-1), (L-3, L-2), ...?
    '''
    Goal:
        apply a list of two site gates in U_list according to the order to sites
        [(0, 1), (1, 2), (2, 3), ... ]
    Input:
        A_list: the MPS representation to which the U_list apply
        U_list: a list of two site unitary gates
        cache: indicate whether the intermediate state should be store.

        no_trunc: indicate whether truncation should take place
        chi: truncate to bond dimension chi, if chi is given.
    Output:
        if cache is True, we will return a list_A_list,
        which gives the list of mps of length L, which corresponds to
        applying 0, 1, 2, ... L-1 gates.
    '''
    L = len(A_list)
    if cache:
        list_A_list = []
        list_A_list.append([A.copy() for A in A_list])

    tot_trunc_err = 0.
    for i in range(L-1):
        d1,chi1,chi2 = A_list[i].shape
        d2,chi2,chi3 = A_list[i+1].shape

        theta = np.tensordot(A_list[i],A_list[i+1],axes=(2,1))
        theta = np.tensordot(U_list[i],theta,axes=([0,1],[0,2]))
        theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))

        X, Y, Z = misc.svd(theta, full_matrices=0)

        if no_trunc:
            chi2 = np.size(Y)
        else:
            chi2 = np.sum(Y>1e-14)

        if chi is not None:
            chi2 = np.amin([chi2, chi])

        trunc_idx = (np.argsort(Y)[::-1])[chi2:]
        trunc_error = np.sum(Y[trunc_idx] ** 2) / np.sum(Y**2)
        tot_trunc_err = tot_trunc_err + trunc_error
        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
        Y = Y[arg_sorted_idx]
        Y = Y / np.linalg.norm(Y)
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        X=np.reshape(X, (d1, chi1, chi2))
        A_list[i]   = X.reshape([d1, chi1, chi2])
        A_list[i+1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])
        if cache:
            list_A_list.append([A.copy() for A in A_list])
        else:
            pass

    if cache:
        return list_A_list, tot_trunc_err
    else:
        return A_list, tot_trunc_err

def apply_U(A_list, U_list, onset):
    '''
    There are two subset of gate.
    onset indicate whether we are applying even (0, 2, 4, ...)
    or odd (1, 3, 5, ...) gates
    '''
    L = len(A_list)

    Ap_list = [None for i in range(L)]
    if L % 2 == 0:
        if onset == 1:
            Ap_list[0] = A_list[0]
            Ap_list[L-1] = A_list[L-1]
        else:
            pass
    else:
        if onset == 0:
            Ap_list[L-1] = A_list[L-1]
        else:
            Ap_list[0] = A_list[0]


    bound = L-1
    for i in range(onset,bound,2):
        d1,chi1,chi2 = A_list[i].shape
        d2,chi2,chi3 = A_list[i+1].shape

        theta = np.tensordot(A_list[i],A_list[i+1],axes=(2,1))
        theta = np.tensordot(U_list[i],theta,axes=([0,1],[0,2]))
        theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))

        X, Y, Z = misc.svd(theta,full_matrices=0)
        chi2 = np.sum(Y>1e-14)

        # piv = np.zeros(len(Y), onp.bool)
        # piv[(np.argsort(Y)[::-1])[:chi2]] = True

        # Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        # X = X[:,piv]
        # Z = Z[piv,:]

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
        Y = Y[arg_sorted_idx]
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        X=np.reshape(X, (d1, chi1, chi2))
        Ap_list[i]   = X.reshape([d1, chi1, chi2])
        Ap_list[i+1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])

    return Ap_list

def var_A(A_list, Ap_list, sweep='left'):
    '''
    ______________ Ap_list = (U|psi>)^\dagger
    |  |  |  |  |

    |  |  |  |  |
    -------------- A_list  = | phi >
    '''
    L = len(A_list)
    # dtype = A_list[0].dtype
    if sweep == 'left':
        Lp = np.ones([1, 1])
        Lp_list = [Lp]

        for i in range(L):
            Lp = np.tensordot(Lp, A_list[i], axes=(0, 1))  #[(1d),1u], [2p,(2l),2r]
            Lp = np.tensordot(Lp, np.conj(Ap_list[i]), axes=([0, 1], [1,0])) #[(1u),(1p),1r], [(2p),(2l),2r]
            Lp_list.append(Lp)

        Rp = np.ones([1, 1])

        A_list_new = [[] for i in range(L)]
        for i in range(L - 1, -1, -1):
            Rp = np.tensordot(Ap_list[i].conj(), Rp, axes=(2, 1))  #[1p,1l,(1r)] [2d,(2u)]
            theta = np.tensordot(Lp_list[i],Rp, axes=(1,1)) #[1d,(1u)], [2p,(2l),2r]
            theta = theta.transpose(1,0,2)  #[lpr]->[plr]
            A_list_new[i] = polar(theta).conj()
            Rp = np.tensordot(A_list_new[i], Rp, axes=([0,2], [0,2]))

        final_overlap = np.einsum('ijk,ijk->', A_list_new[0], theta)
        return A_list_new, final_overlap
    elif sweep == 'right':
        Rp = np.ones([1, 1])
        Rp_list = [Rp]
        for idx in range(L-1, -1, -1):
            Rp = np.tensordot(A_list[idx], Rp, axes=(2, 0))
            Rp = np.tensordot(Rp, np.conj(Ap_list[idx]), axes=([0, 2], [0, 2]))
            Rp_list.append(Rp)

        Lp = np.ones([1, 1])
        A_list_new = [[] for i in range(L)]
        for idx in range(L):
            Lp = np.tensordot(Lp, Ap_list[idx].conj(), axes=(1, 1))
            theta = np.tensordot(Lp, Rp_list[L-1-idx], axes=([2], [1]))
            theta = np.transpose(theta, [1,0,2])
            A_list_new[idx] = polar(theta).conj()
            ## d,ci1,chi2 = theta.shape
            ## Y,D,Z = misc.svd(theta.reshape(d*chi1,chi2), full_matrices=False)
            ## A_list_new[idx] = np.dot(Y,Z).reshape([d,chi1,chi2])
            # print("overlap : ", np.einsum('ijk,ijk->', theta, polar(theta).conj()))
            Lp = np.tensordot(A_list_new[idx], Lp, axes=([0, 1], [1, 0]))

        final_overlap = np.einsum('ijk,ijk->', A_list_new[L-1], theta)
        return A_list_new, final_overlap
    else:
        raise NotImplementedError



