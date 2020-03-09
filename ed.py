""" Exact diagonalization code to find the ground state of
a 1D quantum Ising model."""

import scipy
import scipy.sparse as sparse
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp


def Op_expectation(Op, site_i, vector, L):
    full_Op = scipy.sparse.kron(scipy.sparse.eye(2 ** (site_i)), Op)
    full_Op = scipy.sparse.kron(full_Op, scipy.sparse.eye(2 ** (L - site_i - 1)))
    full_Op = scipy.sparse.csr_matrix(full_Op)
    return vector.conjugate().dot(full_Op.dot(vector))

def gen_spin_operators(L):
    """" Returns the spin operators sigma_x and sigma_z for L sites """
    sx = sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    sy = sparse.csr_matrix(np.array([[0.,-1.j],[1.j,0.]]))
    sz = sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]]))

    d = 2
    sx_list = []
    sy_list = []
    sz_list = []

    for i_site in range(L):
            if i_site==0:
                    X=sx
                    Y=sy
                    Z=sz
            else:
                    X= sparse.csr_matrix(np.eye(d))
                    Y= sparse.csr_matrix(np.eye(d))
                    Z= sparse.csr_matrix(np.eye(d))

            for j_site in range(1,L):
                    if j_site==i_site:
                            X=sparse.kron(X,sx, 'csr')
                            Y=sparse.kron(Y,sy, 'csr')
                            Z=sparse.kron(Z,sz, 'csr')
                    else:
                            X=sparse.kron(X,np.eye(d),'csr')
                            Y=sparse.kron(Y,np.eye(d),'csr')
                            Z=sparse.kron(Z,np.eye(d),'csr')
            sx_list.append(X)
            sy_list.append(Y)
            sz_list.append(Z)

    return sx_list, sy_list, sz_list

def gen_hamiltonian(sx_list, sy_list, sz_list, L):
    """" Generates the Hamiltonian """
    H_xx = sparse.csr_matrix((2**L,2**L))
    H_yy = sparse.csr_matrix((2**L,2**L))
    H_zz = sparse.csr_matrix((2**L,2**L))
    H_x = sparse.csr_matrix((2**L,2**L))
    H_y = sparse.csr_matrix((2**L,2**L))
    H_z = sparse.csr_matrix((2**L,2**L))

    for i in range(L-1):
            H_xx = H_xx + sx_list[i]*sx_list[np.mod(i+1,L)]
            H_yy = H_yy + sy_list[i]*sy_list[np.mod(i+1,L)]
            H_zz = H_zz + sz_list[i]*sz_list[np.mod(i+1,L)]

    for i in range(L):
            H_x = H_x + sx_list[i]
            H_y = H_y + sy_list[i]
            H_z = H_z + sz_list[i]

    return H_xx, H_yy, H_zz, H_x, H_y, H_z

def get_H_Ising(g, J, L):
    sx_list, sy_list, sz_list  = gen_spin_operators(L)
    H_xx, H_yy, H_zz, H_x, H_y, H_z = gen_hamiltonian(sx_list, sy_list, sz_list, L)
    H = J*H_xx + g*H_z
    # H = J*H_zz + g*H_x
    return H

def get_H_XXZ(g, J, L):
    sx_list, sy_list, sz_list  = gen_spin_operators(L)
    H_xx, H_yy, H_zz, H_x, H_y, H_z = gen_hamiltonian(sx_list, sy_list, sz_list, L)
    H = J * (H_xx + H_yy + g * H_zz)
    return H

def get_E_Ising_exact(g,J,L):
    H = get_H_Ising(g, J, L)
    e = arp.eigsh(H,k=1,which='SA',return_eigenvectors=False)
    return(e)

def get_E_XXZ_exact(g,J,L):
    H = get_H_XXZ(g, J, L)
    e = arp.eigsh(H,k=1,which='SA',return_eigenvectors=False)
    return(e)

def get_E_exact(g, J, L, H):
    if H == 'TFI':
        return get_E_Ising_exact(g, J, L)
    elif H == 'XXZ':
        return get_E_XXZ_exact(g, J, L)
    else:
        raise


