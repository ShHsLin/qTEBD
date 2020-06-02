from scipy import integrate
from scipy.linalg import expm
import pickle
import os, sys
sys.path.append('..')
import qTEBD, misc
import numpy as np
# import autograd.numpy as np
# from autograd import grad

def expm_eigh(t, h):
    """
    Compute the unitary operator of a hermitian matrix.
    U = expm(-1j * h)

    Arguments:
    h :: ndarray (N X N) - The matrix to exponentiate, which must be hermitian.

    Returns:
    expm_h :: ndarray(N x N) - The unitary operator of a.
    """
    eigvals, p = np.linalg.eigh(h)
    p_dagger = np.conjugate(np.swapaxes(p, -1, -2))
    d = np.exp(t * eigvals)
    return np.matmul(p *d, p_dagger)

def H_expectation_value(A_list, H_list):
    return np.real(np.sum(qTEBD.expectation_values(A_list, H_list)))

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    h = float(sys.argv[3])
    chi = int(sys.argv[4])
    order = str(sys.argv[5])

    assert order in ['1st', '2nd']
    Hamiltonian = 'TFI'

    ## [TODO] add check whether data already

    J = 1.
    dt = 0.01
    save_each = int(0.5 / dt)
    total_t = 50
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]
    H_list =  qTEBD.get_H(Hamiltonian, L, J, g, h)
    A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    idx = 0

    t_list = [0]
    E_list = [np.sum(qTEBD.expectation_values(A_list, H_list))]
    Sz_array = np.zeros([int(total_t // dt) + 1, L], dtype=np.complex)
    Sz_array[0, :] = qTEBD.expectation_values_1_site(A_list, Sz_list)
    ent_array = np.zeros([int(total_t // dt) + 1, L-1], dtype=np.double)
    ent_array[0, :] = qTEBD.get_entanglement(A_list)
    update_error_list = [0.]

    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)

    wf_dir_path = 'data_tebd/1d_%s_g%.4f_h%.4f/L%d/wf_chi%d_%s/' % (Hamiltonian, g, h, L, chi, order)
    if not os.path.exists(wf_dir_path):
        os.makedirs(wf_dir_path)

    if idx % save_each == 0:
        pickle.dump(A_list, open(wf_dir_path + 'T%.1f.pkl' % t_list[-1],'wb'))

    stop_crit = 1e-1
    for idx in range(1, int(total_t//dt) + 1):
        #######################
        # (apply U_list all)  #
        #######################
        # A_list = qTEBD.right_canonicalize(A_list, no_trunc=True)
        # A_list, trunc_error = qTEBD.apply_U_all(A_list,  U_list, False, no_trunc=False, chi=chi)
        # A_list = qTEBD.left_canonicalize(A_list, no_trunc=True)
        ###########################
        # (apply U_list half half #
        ###########################
        qTEBD.right_canonicalize(A_list, no_trunc=True)
        if order == '2nd':
            Ap_list = qTEBD.apply_U(A_list, U_half_list, 0)
            Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)
            Ap_list = qTEBD.apply_U(Ap_list, U_half_list, 0)
        else:
            Ap_list = qTEBD.apply_U(A_list,  U_list, 0)
            Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)

        qTEBD.right_canonicalize(A_list, no_trunc=True)
        A_list, trunc_error = qTEBD.left_canonicalize(A_list, no_trunc=False, chi=chi)

        ## [ToDo] here assume no truncation
        fidelity_reached = 1. - trunc_error
        print("fidelity reached : ", fidelity_reached)
        update_error_list.append(1. - fidelity_reached)
        current_energy = np.sum(qTEBD.expectation_values(A_list, H_list))
        E_list.append(current_energy)
        Sz_array[idx, :] = qTEBD.expectation_values_1_site(A_list, Sz_list)
        ent_array[idx, :] = qTEBD.get_entanglement(A_list)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])
        print("current chi : ", A_list[L//2].shape[1])

        if idx % save_each == 0:
            pickle.dump(A_list, open(wf_dir_path + 'T%.1f.pkl' % t_list[-1],'wb'))

        ################
        ## Forcing to stop if truncation is already too high.
        ################
        total_trunc_error = np.abs(1 - np.multiply.reduce(1. - np.array(update_error_list)))
        if total_trunc_error > stop_crit:
            break

    num_data = len(t_list)
    Sz_array = Sz_array[:num_data, :]
    ent_array = ent_array[:num_data, :]

    dir_path = 'data_tebd/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'mps_chi%d_%s_energy.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(E_list))

    filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(t_list))

    filename = 'mps_chi%d_%s_error.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(update_error_list))

    filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, Sz_array)

    filename = 'mps_chi%d_%s_ent_array.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, ent_array)

