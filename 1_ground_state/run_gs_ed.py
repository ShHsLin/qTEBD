""" Exact diagonalization code to find the ground state of
a 1D quantum Ising model."""
import sys
sys.path.append('..')
import scipy.sparse as sparse
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp
import ed

if __name__ == "__main__":
    import sys, os
    import misc
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    J = 1.

    H = 'XXZ'  # XXZ

    if H == 'TFI':
        dir_path = 'data/1d_TFI_g%.1f/' % (g)
    elif H == 'XXZ':
        dir_path = 'data/1d_XXZ_g%.1f/' % (g)

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
        energy = ed.get_E_exact(g, J, L, H)
        E_dict[L] = energy
        misc.save_array(path, misc.dict_2_nparray(E_dict))
        # If no data --> generate data
        print("Save new data")


