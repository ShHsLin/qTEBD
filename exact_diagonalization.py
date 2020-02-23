""" Exact diagonalization code to find the ground state of 
a 1D quantum Ising model."""

import scipy.sparse as sparse 
import numpy as np 
import scipy.sparse.linalg.eigen.arpack as arp
import pylab as pl

def gen_spin_operators(L): 
	"""" Returns the spin operators sigma_x and sigma_z for L sites """
	sx = sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]))
	sz = sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]]))
	    
	d = 2
	sx_list = []
	sz_list = []

	for i_site in range(L): 
		if i_site==0: 
			X=sx
			Z=sz 
		else: 
			X= sparse.csr_matrix(np.eye(d)) 
			Z= sparse.csr_matrix(np.eye(d))
            
		for j_site in range(1,L): 
			if j_site==i_site: 
				X=sparse.kron(X,sx, 'csr')
				Z=sparse.kron(Z,sz, 'csr') 
			else: 
				X=sparse.kron(X,np.eye(d),'csr') 
				Z=sparse.kron(Z,np.eye(d),'csr') 
		sx_list.append(X)
		sz_list.append(Z) 
		
	return sx_list,sz_list 

def gen_hamiltonian(sx_list,sz_list,L): 
	"""" Generates the Hamiltonian """    
	H_zz = sparse.csr_matrix((2**L,2**L))
	H_x = sparse.csr_matrix((2**L,2**L))       
	    
	for i in range(L-1):
		H_zz = H_zz + sz_list[i]*sz_list[np.mod(i+1,L)]
	for i in range(L):
		H_x = H_x + sx_list[i]
     
	return H_zz, H_x 
	
def get_E_Ising_exact(g,J,L):
	sx_list,sz_list  = gen_spin_operators(L)
	H_zz, H_x = gen_hamiltonian(sx_list,sz_list,L)
	H = J*H_zz + g*H_x
	e = arp.eigsh(H,k=1,which='SA',return_eigenvectors=False)
	return(e)
		
		
if __name__ == "__main__":
    import sys, os
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    J = 1.

    import misc
    dir_path = 'data/1d_TFI_g%.1f/' % (g)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'exact_energy.csv'
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
        energy = get_E_Ising_exact(g, J, L)
        E_dict[L] = energy
        misc.save_array(path, misc.dict_2_nparray(E_dict))
        # If no data --> generate data
        print("Save new data")


