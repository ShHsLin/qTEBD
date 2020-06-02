import numpy as np

def MPS_2_state(mps):
    '''
    Goal:
        Return the full tensor representation (vector) of the state
    Input:
        MPS
    '''
    Vec = mps[0][:,0,:]
    for idx in range(1, len(mps)):
        Vec = np.einsum('pa,qal->pql', Vec, mps[idx])
        dim_p, dim_q, dim_l = Vec.shape
        Vec = Vec.reshape([dim_p * dim_q, dim_l])

    return Vec.flatten()


def lpr_2_plr(A_list):
    return [np.transpose(a, [1,0,2]) for a in A_list]

def plr_2_lpr(A_list):
    return [np.transpose(a, [1,0,2]) for a in A_list]

def MPS_dot_left_env(mps_up, mps_down, site_l, cache_env_list=None):
    '''
    # Complex compatible
    Goal:
        Contract and form the left environment of site_l
    Input:
        mps_up : up (should actually add conjugate !!! )
        mps_down : down
        site_l : Index convention starting from 0
    Output:
        left_environment

    |-----------------  -----
    | left                |
    | environment         |site_l
    |-----------------  -----
    0,...,(site_l-1)
    '''
    if site_l == 0:
        return np.eye(1)

    left_env = np.eye(1)
    for idx in range(0, site_l):
        left_env = np.einsum('ij,ikl->jkl', left_env, mps_up[idx].conjugate())
        left_env = np.einsum('ijk,ijl->kl', left_env, mps_down[idx])
        if not (cache_env_list is None):
            cache_env_list[idx] = left_env

    return left_env


def MPS_dot_right_env(mps_up, mps_down, site_l, cache_env_list=None):
    '''
    # Complex compatible
    Goal:
        Contract and form the right environment of site_l
    Input:
        mps_up : up (should actually add conjugate !!! )
        mps_down : down
        site_l : Index convention starting from 0
        cache_env_list : List with length L
    Output:
        right_environment
        cache_env_list : with list[site_l] = contraction including site_l

      -----       -----------------|
        |          right           |
        |site_l    environment     |
      -----       -----------------|
                    (site_l+1),...,L-1
    '''
    L = len(mps_up)
    if site_l == L - 1:
        return np.eye(1)

    right_env = np.eye(1)
    for idx in range(L - 1, site_l, -1):
        right_env = np.einsum('kli, ij ->klj', mps_up[idx].conjugate(),
                              right_env)
        right_env = np.einsum('klj,mlj->km', right_env, mps_down[idx])
        if not (cache_env_list is None):
            cache_env_list[idx] = right_env

    return right_env


def MPS_dot(mps_1, mps_2):
    '''
    # Complex compatible
    Return inner product of two MPS, with mps_1 taking complex_conjugate
    <mps_1 | mps_2 >
    '''
    L = len(mps_1)
    mps_temp = np.einsum('ijk,ijl->kl', mps_1[0].conjugate(), mps_2[0])
    for idx in range(1, L):
        mps_temp = np.einsum('ij,ikl->jkl', mps_temp, mps_1[idx].conjugate())
        mps_temp = np.einsum('ijk,ijl->kl', mps_temp, mps_2[idx])

    return mps_temp[0, 0]


def MPS_compression_variational(mps_trial, mps_target, max_iter=30, tol=1e-4,
                                verbose=0):
    '''
    Variational Compression on MPS with mps_trial given.
    Input:
        mps_trial: MPS for optimization
            The input should be in right canonical form BBBBBB and
            it should be normalized.
        mps_target: The target to approximate.
            It is not necessary in canonical form and not necessarily
            normalized.

    Output:
        trunc_err
        modification mps_trial inplace still in right canonical form
    '''
    L = len(mps_trial)
    # Check normalization
    if np.abs(MPS_dot(mps_trial, mps_trial) - 1.) > 1e-8:
        print(('mps_comp_var not normalized', MPS_dot(mps_trial, mps_trial)))
        raise
    elif np.abs(MPS_dot(mps_target, mps_target) - 1.) > 1e-8:
        print(('mps_comp_var not normalized', MPS_dot(mps_target, mps_target)))
        mps_target[-1] /= np.sqrt(MPS_dot(mps_target, mps_target))
    else:
        pass
        # all normalized

    conv = False
    num_iter = 0
    old_trunc_err = 1.
    while (num_iter < max_iter and not conv):
        num_iter += 1
        # Creat cache of environment
        cache_env_list = [None] * L
        MPS_dot_right_env(mps_trial,
                          mps_target,
                          0,
                          cache_env_list=cache_env_list)

        # site = 0
        right_env = cache_env_list[1]
        left_env = np.eye(1)
        update_tensor = np.einsum('ij,jkl->ikl', left_env, mps_target[0])
        update_tensor = np.einsum('ikl,ml->ikm', update_tensor, right_env)
        mps_trial[0] = update_tensor
        # svd to shift central site
        l_dim, d, r_dim = mps_trial[0].shape
        U, s, Vh = np.linalg.svd(mps_trial[0].reshape((l_dim * d, r_dim)),
                                 full_matrices=False)
        rank = s.size
        s /= np.linalg.norm(s)
        mps_trial[0] = U.reshape((l_dim, d, rank))
        mps_trial[1] = np.einsum('ij,jkl->ikl',
                                 np.diag(s).dot(Vh), mps_trial[1])
        # update env
        left_env = np.einsum('ijk,ijl->kl', mps_trial[0].conjugate(),
                             mps_target[0])
        cache_env_list[0] = left_env
        cache_env_list[1] = None

        for site in range(1, L - 1):
            right_env = cache_env_list[site + 1]
            left_env = cache_env_list[site - 1]
            update_tensor = np.einsum('ij,jkl->ikl', left_env,
                                      mps_target[site])
            update_tensor = np.einsum('ikl,ml->ikm', update_tensor, right_env)
            mps_trial[site] = update_tensor
            # svd to shift central site
            l_dim, d, r_dim = mps_trial[site].shape
            U, s, Vh = np.linalg.svd(mps_trial[site].reshape(
                (l_dim * d, r_dim)),
                                     full_matrices=False)
            rank = s.size
            s /= np.linalg.norm(s)
            mps_trial[site] = U.reshape((l_dim, d, rank))
            mps_trial[site + 1] = np.einsum('ij,jkl->ikl',
                                            np.diag(s).dot(Vh),
                                            mps_trial[site + 1])
            # update env
            left_env = np.einsum('ij,ikl->jkl', left_env,
                                 mps_trial[site].conjugate())
            left_env = np.einsum('ijk,ijl->kl', left_env, mps_target[site])
            cache_env_list[site] = left_env
            cache_env_list[site + 1] = None

        # site = L-1
        right_env = np.eye(1)
        left_env = cache_env_list[L - 2]
        update_tensor = np.einsum('ij,jkl->ikl', left_env, mps_target[L - 1])
        update_tensor = np.einsum('ikl,ml->ikm', update_tensor, right_env)
        mps_trial[L - 1] = update_tensor
        # svd to shift central site
        l_dim, d, r_dim = mps_trial[L - 1].shape
        U, s, Vh = np.linalg.svd(mps_trial[L - 1].reshape((l_dim, d * r_dim)),
                                 full_matrices=False)
        rank = s.size
        s /= np.linalg.norm(s)
        mps_trial[L - 1] = Vh.reshape((rank, d, r_dim))
        mps_trial[L - 2] = np.einsum('ijk,kl->ijl', mps_trial[L - 2],
                                     U.dot(np.diag(s)))
        # No update for left_env

        # site = L-1
        # But update for right_env
        right_env = np.einsum('ijk,ljk->il', mps_trial[L - 1].conjugate(),
                              mps_target[L - 1])
        cache_env_list[L - 1] = right_env
        cache_env_list[L - 2] = None

        for site in range(L - 2, 0, -1):
            right_env = cache_env_list[site + 1]
            left_env = cache_env_list[site - 1]
            update_tensor = np.einsum('ij,jkl->ikl', left_env,
                                      mps_target[site])
            update_tensor = np.einsum('ikl,ml->ikm', update_tensor, right_env)
            mps_trial[site] = update_tensor
            # svd to shift central site
            l_dim, d, r_dim = mps_trial[site].shape
            U, s, Vh = np.linalg.svd(mps_trial[site].reshape(
                (l_dim, d * r_dim)),
                                     full_matrices=False)
            rank = s.size
            s /= np.linalg.norm(s)
            mps_trial[site] = Vh.reshape((rank, d, r_dim))
            mps_trial[site - 1] = np.einsum('ijk,kl->ijl', mps_trial[site - 1],
                                            U.dot(np.diag(s)))
            # update env
            right_env = np.einsum('kli, ij ->klj', mps_trial[site].conjugate(),
                                  right_env)
            right_env = np.einsum('klj,mlj->km', right_env, mps_target[site])
            cache_env_list[site] = right_env
            cache_env_list[site - 1] = None

        # site = 0
        trunc_err = 1. - np.square(np.abs(MPS_dot(mps_trial, mps_target)))
        if verbose:
            print(('var_trunc_err = ', trunc_err))

        if np.abs(old_trunc_err - trunc_err) < 1e-6:
            conv = True

        old_trunc_err = trunc_err

    return trunc_err

