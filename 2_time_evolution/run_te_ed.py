"""
Exact diagonalization code to do exact time evolution for
1D quantum Ising model and XXZ model.
"""

import scipy.sparse as sparse
import scipy.sparse.linalg
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp
import sys
sys.path.append('..')
import ed




if __name__ == "__main__":
    import sys, os
    import misc
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    h = float(sys.argv[3])
    J = 1.

    H = 'TFI'

    if H == 'TFI':
        dir_path = 'data_te/1d_TFI_g%.4f_h%.4f/L%d/' % (g, h, L)
        # H = xx + gz
        H = ed.get_H_Ising(g, h, J, L)
    elif H == 'XXZ':
        dir_path = 'data_te/1d_XXZ_g%.1f/L%d/' % (g, L)
        H = ed.get_H_XXZ(g, J, L)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    ######   1111111111 STATE     ######
    up_state = np.array([1., 0.])
    init_state = np.array(1.)
    for i in range(L):
        init_state = np.kron(init_state, up_state)

    ######   XXXXXXXXX STATE     ######
    # init_state = np.ones( 2 ** L, dtype=np.complex128)
    # init_state = init_state / np.linalg.norm(init_state)


    Sz = np.array([[1., 0.],
                   [0., -1.]])
    Sx = np.array([[0., 1.],
                   [1., 0.]])
    ####### Evolve state #########
    psi = init_state
    dt = 0.01
    total_time = 30.
    sz_array = np.zeros([int(total_time // dt) + 1, L])
    for site in range(L):
        sz_array[0, site] = ed.Op_expectation(Sz, L//2, psi, L)

    for i in range(1, int(total_time // dt) +  1):
        psi = scipy.sparse.linalg.expm_multiply(-1.j*dt*H, psi)
        for site in range(L):
            sz_array[i, site] = ed.Op_expectation(Sz, L//2, psi, L)

        print("<E(%.2f)> : " % (i*dt), psi.conj().T.dot(H.dot(psi)), "sz:", sz_array[i, L//2])


    filename = 'ed_dt_array.npy'
    path = dir_path + filename
    np.save(path, np.arange(int(total_time // dt) +  1) * dt)

    filename = 'ed_sz_array.npy'
    path = dir_path + filename
    np.save(path, sz_array)

