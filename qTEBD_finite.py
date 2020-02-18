from scipy import integrate
from scipy.linalg import expm
import numpy as np
import pylab as pl
import exact_diagonalization as ed

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
        return (-np.kron(sz, sz) * J - gr * np.kron(id, sx) - gl * np.kron(sx, id)).reshape([d] * 4)
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

def overlap(psi1,psi2):
    N = np.ones([1,1]) # a ap
    L = len(psi1)
    for i in np.arange(L):
        N = np.tensordot(N,np.conj(psi1[i]), axes=(1,1)) # a ap 
        N = np.tensordot(N,psi2[i], axes=([0,1],[1,0])) # ap a
        N = N.transpose(1,0)
    N = np.trace(N)
    return(N)

def expectation_values(A_list, H_list):
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
        Rp = np.tensordot(A_list[i+1].conj(),Rp, axes=(2, 1))
        E = np.tensordot(A_list[i].conj(),Rp, axes=(2, 1))
        E = np.tensordot(H_list[i],E, axes=([0,1], [0,2]))
        E = np.tensordot(A_list[i+1],E, axes=([0,2], [1,3]))
        E = np.tensordot(A_list[i],E, axes=([0,2], [1,0]))
        E = np.tensordot(Lp_list[i],E, axes=([0,1],[0,1]))
        Rp = np.tensordot(A_list[i+1],Rp,axes=([0,2], [0,2]))
        E_list.append(E.item())
    return E_list

def apply_U(A_list, U_list, onset):
    L = len(A_list)
    
    Ap_list = [[] for i in range(L)]
    if onset == 1:
        Ap_list[0] = A_list[0]
        Ap_list[L-1] = A_list[L-1]
         
    for i in range(onset,L-1,2):
        d1,chi1,chi2 = A_list[i].shape
        d2,chi2,chi3 = A_list[i+1].shape
        
        theta = np.tensordot(A_list[i],A_list[i+1],axes=(2,1))
        theta = np.tensordot(U_list[i],theta,axes=([0,1],[0,2]))
        theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1,d2*chi3))

        X, Y, Z = np.linalg.svd(theta,full_matrices=0)
        chi2 = np.sum(Y>10.**(-10))

        piv = np.zeros(len(Y), np.bool)
        piv[(np.argsort(Y)[::-1])[:chi2]] = True

        Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        X = X[:,piv] 
        Z = Z[piv,:]

        X=np.reshape(X,(d1,chi1,chi2))
        Ap_list[i]   = X.reshape([d1,chi1,chi2])
        Ap_list[i+1] = np.dot(np.diag(Y),Z).reshape([chi2,d2,chi3]).transpose([1,0,2])

    return(Ap_list) 

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
    
if __name__ == "__main__":
    np.random.seed(1)  
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
    L = 10
    chi = 4
    J = 1.
    g = 1.5
    N_iter = 1
    
    A_list  =  init_mps(L,chi,2)
    H_list  =  get_H_TFI(L, J, g)
    E_exact =  ed.get_E_Ising_exact(g,J,L)
    delta_list = [np.sum(expectation_values(A_list, H_list))-E_exact.item()]
    t_list = [0]
    for dt in [0.05,0.01,0.001]:
        U_list =  make_U(H_list, dt)
        for i in range(int(20//dt**(0.75))):
            Ap_list = apply_U(A_list,  U_list, 0)
            Ap_list = apply_U(Ap_list, U_list, 1)
            
            for a in range(N_iter):
                A_list  = var_A(A_list, Ap_list)
            
            delta_list.append(np.sum(expectation_values(A_list, H_list))-E_exact.item())
            t_list.append(t_list[-1]+dt)
            
            print(t_list[-1],delta_list[-1])
    
    pl.close()
    pl.semilogy(t_list,delta_list,'.')
    pl.title('$\\chi=$'+str(chi) + ', $L=$' + str(L))
    pl.xlabel('$\\tau$')
    pl.ylabel('$E_{\\tau} - E_0$')
    pl.legend(['$\\chi=%.0f$'%chi])
    pl.savefig('plot.pdf')
    pl.show()
