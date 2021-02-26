from scipy import integrate
from scipy.linalg import expm
# import numpy as np
import os, sys
sys.path.append('..')
import qTEBD, misc
import autograd.numpy as np
from autograd import grad

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
    return np.real(np.sum(mps_func.expectation_values(A_list, H_list)))

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
    tol = 1e-14
    cov_crit = tol * 0.1
    max_N_iter = 100
    N_iter = 10
    dt = 0.01
    total_t = 30
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]
    H_list =  qTEBD.get_H(Hamiltonian, L, J, g, h)
    A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]

    t_list = [0]
    E_list = [np.sum(mps_func.expectation_values(A_list, H_list))]
    Sz_array = np.zeros([int(total_t // dt) + 1, L], dtype=np.complex)
    Sz_array[0, :] = mps_func.expectation_values_1_site(A_list, Sz_list)
    ent_array = np.zeros([int(total_t // dt) + 1, L-1], dtype=np.double)
    ent_array[0, :] = mps_func.get_entanglement(A_list)
    update_error_list = [0.]

    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)

    exact_steps = int(np.log2(chi))
    for idx in range(exact_steps):
        A_list, _ = mps_func.right_canonicalize(A_list, no_trunc=True)
        A_list, trunc_error = qTEBD.apply_U_all(A_list,  U_list, 0, no_trunc=True)
        A_list, _ = mps_func.left_canonicalize(A_list, no_trunc=True)

        ## [ToDo] here assume no truncation
        fidelity_reached = 1.
        print("fidelity reached : ", fidelity_reached)
        update_error_list.append(1. - fidelity_reached)
        current_energy = np.sum(mps_func.expectation_values(A_list, H_list))
        E_list.append(current_energy)
        Sz_array[1+idx, :] = mps_func.expectation_values_1_site(A_list, Sz_list)
        ent_array[1+idx, :] = mps_func.get_entanglement(A_list)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])
        print("current chi : ", A_list[L//2].shape[1])

    stop_crit = 1e-4
    first_break_idx = np.inf
    for idx in range(1+exact_steps, int(total_t//dt) + 1):
        if fidelity_reached < 1. - 1e-12 and A_list[L//2].shape[1] < chi:
            ### AAAAA form
            A_list, _ = mps_func.right_canonicalize(A_list)
            ### BBBBB form
            A_list, trunc_error = qTEBD.apply_U_all(A_list,  U_list, 0)
            ### AAAAA form

            ## [ToDo] here assume no truncation
            fidelity_reached = 1.
            print("fidelity reached : ", fidelity_reached)
            update_error_list.append(1. - fidelity_reached)
            current_energy = np.sum(mps_func.expectation_values(A_list, H_list))
            E_list.append(current_energy)
            Sz_array[idx, :] = mps_func.expectation_values_1_site(A_list, Sz_list)
            ent_array[idx, :] = mps_func.get_entanglement(A_list)
            t_list.append(t_list[-1]+dt)

            print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])
            # N_iter = 10
            continue

        if order == '2nd':
            Ap_list = qTEBD.apply_U(A_list, U_half_list, 0)
            Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)
            Ap_list = qTEBD.apply_U(Ap_list, U_half_list, 0)
        else:
            Ap_list = qTEBD.apply_U(A_list,  U_list, 0)
            Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)

        # Ap_list = e^(-i dt H) | A_list >
        # print("Norm new mps = ", mps_func.overlap(Ap_list, Ap_list), "new state aimed E = ",
        #       np.sum(mps_func.expectation_values(Ap_list, H_list, check_norm=False))/mps_func.overlap(Ap_list, Ap_list)
        #      )

        ### POLAR DECOMPOSITION UPDATE ###
        Ap_norm_sq = mps_func.overlap(Ap_list, Ap_list)
        fidelity_before = np.abs(mps_func.overlap(Ap_list, A_list))**2 / Ap_norm_sq
        print("fidelity before : ", fidelity_before)

        A_list, overlap  = qTEBD.var_A(A_list, Ap_list, 'left')
        F = np.abs(overlap) ** 2 / Ap_norm_sq
        num_iter = 0
        F_diff = 1
        while ( num_iter < max_N_iter and np.abs(1-F) > tol and F_diff > cov_crit):
            num_iter = num_iter + 1
            print("sweeping right")
            A_list, overlap  = qTEBD.var_A(A_list, Ap_list, 'right')
            print("sweeping left")
            A_list, overlap  = qTEBD.var_A(A_list, Ap_list, 'left')
            new_F = np.abs(overlap) ** 2 / Ap_norm_sq
            F_diff = np.abs(new_F - F)
            F = new_F
            print(" F = ", F, "F_diff", F_diff)


        # else:
        #     ## Gradient opt
        #     for k in range(100):
        #         grad_A_list = grad(mps_func.overlap, 1)(Ap_list, A_list)
        #         for idx_2 in range(L):
        #             d, chi1, chi2 = A_list[idx_2].shape
        #             U = A_list[idx_2].reshape([d * chi1, chi2])
        #             dU = grad_A_list[idx_2].reshape([d * chi1, chi2])
        #             U, dU = U.T, dU.T
        #             M = U.T.conj().dot(dU) - dU.T.conj().dot(U)
        #             U_update = U.dot(expm_eigh(+dt*0.01, M))
        #             U, dU, U_update = U.T, dU.T, U_update.T
        #             # import pdb;pdb.set_trace()
        #             A_list[idx_2] = U_update.reshape([d, chi1, chi2])
        #             # A_list[idx_2] = A_list[idx_2] + dt * grad_A_list[idx_2]

        #     for a in range(N_iter):
        #         A_list, overlap  = qTEBD.var_A(A_list, A_list, 'right')

        # ############################################
        # ### Hamiltonian expectation minimization ###
        # ############################################
        # # if t_list[-1] >= 18:
        # #     grad_A_list = grad(H_expectation_value, 0)(A_list, H_list)
        # #     for idx_2 in range(L):
        # #         d, chi1, chi2 = A_list[idx_2].shape
        # #         U = A_list[idx_2].reshape([d * chi1, chi2])
        # #         dU = grad_A_list[idx_2].reshape([d * chi1, chi2])
        # #         U, dU = U.T, dU.T
        # #         M = U.T.conj().dot(dU) - dU.T.conj().dot(U)
        # #         U_update = U.dot(expm_eigh(dt*0.01, M))
        # #         U, dU, U_update = U.T, dU.T, U_update.T
        # #         # import pdb;pdb.set_trace()
        # #         A_list[idx_2] = U_update.reshape([d, chi1, chi2])
        # #         # A_list[idx_2] = A_list[idx_2] + dt * grad_A_list[idx_2]

        # #     for a in range(N_iter):
        # #         A_list, overlap  = qTEBD.var_A(A_list, A_list, 'right')

        fidelity_reached = np.abs(mps_func.overlap(Ap_list, A_list))**2 / mps_func.overlap(Ap_list, Ap_list)
        print("fidelity reached : ", fidelity_reached)
        update_error_list.append(1. - fidelity_reached)
        current_energy = np.sum(mps_func.expectation_values(A_list, H_list))
        E_list.append(current_energy)
        Sz_array[idx, :] = mps_func.expectation_values_1_site(A_list, Sz_list)
        ent_array[idx, :] = mps_func.get_entanglement(A_list)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])

        trunc_error = np.abs(1. - fidelity_reached)
        if trunc_error > stop_crit:
            first_break_idx = np.amin([first_break_idx, idx])

        if first_break_idx + int(1.//dt) < idx:
            break

    num_data = len(t_list)
    Sz_array = Sz_array[:num_data, :]
    ent_array = ent_array[:num_data, :]

    dir_path = 'data_te/1d_%s_g%.4f_h%.4f/L%d/' % (Hamiltonian, g, h, L)
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

