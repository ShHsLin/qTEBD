import numpy as np
import pickle
import os, sys
sys.path.append('..')
import qTEBD, misc, mps_func

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5,threshold=4000)
    L = int(sys.argv[1])
    J = 1.
    g = float(sys.argv[2])
    h = float(sys.argv[3])
    data_order = str(sys.argv[4])
    new_chi = int(sys.argv[5])
    try:
        data_chi = int(sys.argv[6])
    except:
        data_chi = 128

    data_dt = 1e-3

    assert data_order in ['1st', '2nd', '4th']

    Hamiltonian = 'TFI'
    H_list  =  qTEBD.get_H(Hamiltonian, L, J, g, h)
    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]


    total_steps = 1000
    E_list = []
    Sz_array = np.zeros([total_steps, L])
    ent_array = np.zeros([total_steps, L-1])
    error_list = []
    t_list = []

    for T_idx in range(1,total_steps):
        T = T_idx * 0.5

        ############### LOAD TARGET STATE ######################
        try:
            mps_dir_path = 'data_tebd_dt%e/1d_%s_g%.4f_h%.4f/L%d/wf_chi%d_%s/' % (data_dt, Hamiltonian, g, h, L, data_chi, data_order)
            filename = mps_dir_path + 'T%.1f.pkl' % T
            target_mps = pickle.load(open(filename, 'rb'))
        except Exception as e:
            print(e)
            break

        ############### SET UP INITIALIZATION #################
        # target_mps in left canonical form
        trunc_mps = [t.copy() for t in target_mps]
        mps_func.right_canonicalize(trunc_mps, no_trunc=True)
        trunc_mps, trunc_error = mps_func.left_canonicalize(trunc_mps, no_trunc=False, chi=new_chi)
        mps_func.right_canonicalize(trunc_mps, no_trunc=True)
        print("trunc_error : ", trunc_error)
        target_mps_ = mps_func.plr_2_lpr(target_mps)
        trunc_mps_ = mps_func.plr_2_lpr(trunc_mps)
        trunc_error_ = mps_func.MPS_compression_variational(trunc_mps_, target_mps_, verbose=1)
        print("trunc_error (var) : ", trunc_error_)

        trunc_mps = mps_func.lpr_2_plr(trunc_mps_)
        mps_func.left_canonicalize(trunc_mps, no_trunc=True)


        E_list.append(np.sum(mps_func.expectation_values(trunc_mps, H_list)))
        Sz_array[T_idx, :] = mps_func.expectation_values_1_site(trunc_mps, Sz_list)
        ent_array[T_idx, :] = mps_func.get_entanglement(trunc_mps)
        # print("ENTANGLEMENT !!!!! : ", mps_func.get_entanglement(trunc_mps))
        fidelity_reached = np.abs(mps_func.overlap(target_mps, trunc_mps))**2
        error_list.append(1. - fidelity_reached)
        t_list.append(T)

        # trunc_mps_dir_path = 'data_tebd_dt%e/1d_%s_g%.4f_h%.4f/L%d/trunc_wf_chi%d_%s/' % (data_dt, Hamiltonian, g, h, L, new_chi, data_order)
        # if not os.path.exists(trunc_mps_dir_path):
        #     os.makedirs(trunc_mps_dir_path)

        # filename = trunc_mps_dir_path + 'T%.1f.pkl' % T
        # pickle.dump(trunc_mps, open(filename, 'wb'))

    num_data = len(t_list)
    assert num_data > 0
    Sz_array = Sz_array[:num_data, :]
    ent_array = ent_array[:num_data, :]


    dir_path = 'data/1d_%s_g%.4f_h%.4f/L%d_chi%d/approx_mps/' % (Hamiltonian, g, h, L, data_chi)
    # wf_dir_path = dir_path + 'wf_depth%d_Niter%d_%s/' % (depth, N_iter, order)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'mps_chi%d_%s_energy.npy' % (new_chi, data_order)
    path = dir_path + filename
    np.save(path, np.array(E_list))

    filename = 'mps_chi%d_%s_dt.npy' % (new_chi, data_order)
    path = dir_path + filename
    np.save(path, np.array(t_list))

    filename = 'mps_chi%d_%s_error.npy' % (new_chi, data_order)
    path = dir_path + filename
    np.save(path, np.array(error_list))

    filename = 'mps_chi%d_%s_sz_array.npy' % (new_chi, data_order)
    path = dir_path + filename
    np.save(path, Sz_array)

    filename = 'mps_chi%d_%s_ent_array.npy' % (new_chi, data_order)
    path = dir_path + filename
    np.save(path, ent_array)


