from scipy import integrate
from scipy.linalg import expm
# import numpy as np
import misc, os, sys
import qTEBD
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
    return np.real(np.sum(qTEBD.expectation_values(A_list, H_list)))

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
    L = int(sys.argv[1])
    g = float(sys.argv[2])
    chi = int(sys.argv[3])
    order = str(sys.argv[4])

    assert order in ['1st', '2nd']
    Hamiltonian = 'TFI'

    ## [TODO] add check whether data already

    J = 1.
    N_iter = 1
    dt = 0.01
    total_t = 3
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]
    H_list =  qTEBD.get_H(L, J, g, Hamiltonian)
    A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]

    t_list = [0]
    E_list = [np.sum(qTEBD.expectation_values(A_list, H_list))]
    Sz_array = np.zeros([int(total_t // dt) + 1, L])
    Sz_array[0, :] = qTEBD.expectation_values_1_site(A_list, Sz_list)
    update_error_list = [0.]

    U_list =  qTEBD.make_U(H_list, 1j * dt)
    U_half_list =  qTEBD.make_U(H_list, 0.5j * dt)

    exact_steps = int(np.log2(chi))
    exact_steps = 1
    for idx in range(exact_steps):
        A_list = qTEBD.right_canonicalize(A_list)
        A_list = qTEBD.apply_U_all(A_list,  U_list, 0)
        A_list = qTEBD.left_canonicalize(A_list)

        ## [ToDo] here assume no truncation
        fidelity_reached = 1.
        print("fidelity reached : ", fidelity_reached)
        update_error_list.append(1. - fidelity_reached)
        current_energy = np.sum(qTEBD.expectation_values(A_list, H_list))
        E_list.append(current_energy)
        Sz_array[1+idx, :] = qTEBD.expectation_values_1_site(A_list, Sz_list)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])

    for idx in range(1+exact_steps, int(total_t//dt) + 1):
        if fidelity_reached < 1. - 1e-3 and A_list[L//2].shape[1] < chi:
            A_list = qTEBD.right_canonicalize(A_list)
            A_list = qTEBD.apply_U_all(A_list, [np.eye(4).reshape([2]*4)]*(L-1), 0)
            # ### AAAAA form
            # A_list = qTEBD.right_canonicalize(A_list)
            # ### BBBBB form
            # A_list = qTEBD.apply_U_all(A_list,  U_list, 0)
            # ### AAAAA form
            # # for a in range(10):
            # #     A_list  = qTEBD.var_A(A_list, Ap_list, 'right')

            # ## [ToDo] here assume no truncation
            # fidelity_reached = 1.
            # print("fidelity reached : ", fidelity_reached)
            # update_error_list.append(1. - fidelity_reached)
            # current_energy = np.sum(qTEBD.expectation_values(A_list, H_list))
            # E_list.append(current_energy)
            # Sz_array[idx, :] = qTEBD.expectation_values_1_site(A_list, Sz_list)
            # t_list.append(t_list[-1]+dt)

            # print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])
            # N_iter = 10
            # continue

        if order == '2nd':
            Ap_list = qTEBD.apply_U(A_list, U_half_list, 0)
            Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)
            Ap_list = qTEBD.apply_U(Ap_list, U_half_list, 0)
        else:
            Ap_list = qTEBD.apply_U(A_list,  U_list, 0)
            Ap_list = qTEBD.apply_U(Ap_list, U_list, 1)

        # Ap_list = e^(-i dt H) | A_list >
        # print("Norm new mps = ", qTEBD.overlap(Ap_list, Ap_list), "new state aimed E = ",
        #       np.sum(qTEBD.expectation_values(Ap_list, H_list, check_norm=False))/qTEBD.overlap(Ap_list, Ap_list)
        #      )

        Ap_list = qTEBD.left_canonicalize(Ap_list)
        ### POLAR DECOMPOSITION UPDATE ###
        fidelity_before = np.abs(qTEBD.overlap(Ap_list, A_list))**2 / qTEBD.overlap(Ap_list, Ap_list)
        print("fidelity before : ", fidelity_before)
        for a in range(N_iter):
            # A_list  = qTEBD.var_A(A_list, Ap_list, 'left')
            A_list  = qTEBD.var_A(A_list, Ap_list, 'right')


        # else:
        #     ## Gradient opt
        #     for k in range(100):
        #         grad_A_list = grad(qTEBD.overlap, 1)(Ap_list, A_list)
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
        #         A_list  = qTEBD.var_A(A_list, A_list, 'right')

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
        # #         A_list  = qTEBD.var_A(A_list, A_list, 'right')

        fidelity_reached = np.abs(qTEBD.overlap(Ap_list, A_list))**2 / qTEBD.overlap(Ap_list, Ap_list)
        print("fidelity reached : ", fidelity_reached)
        update_error_list.append(1. - fidelity_reached)
        current_energy = np.sum(qTEBD.expectation_values(A_list, H_list))
        E_list.append(current_energy)
        Sz_array[idx, :] = qTEBD.expectation_values_1_site(A_list, Sz_list)
        t_list.append(t_list[-1]+dt)

        print("T=", t_list[-1], " E=", E_list[-1], " Sz=", Sz_array[idx, L//2])

    dir_path = 'data_te/1d_%s_g%.1f/L%d/' % (Hamiltonian, g, L)
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

