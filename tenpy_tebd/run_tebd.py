
import numpy as np
import pickle
import os, sys
sys.path.append('../')


from tenpy.networks.mps import MPS
from tenpy.algorithms import tebd


import numpy as np

from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import get_parameter
from tenpy.networks.site import SpinHalfSite
import mps_func


class IsingModel(CouplingMPOModel):
    r"""General Ising model on a general lattice.

    The Hamiltonian reads:

    .. math ::
        H = - \sum_{\langle i,j\rangle, i < j} \mathtt{J} \sigma^x_i \sigma^x_{j}
            - \sum_{i} \mathtt{g} \sigma^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs, each pair appearing
    exactly once.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`TFIModel` below.

    Options
    -------
    .. cfg:config :: TFIModel
        :include: CouplingMPOModel

        conserve : None | 'parity'
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
        J, g : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        # conserve = get_parameter(model_params, 'conserve', 'parity', self.name)
        # assert conserve != 'Sz'
        # if conserve == 'best':
        #     conserve = 'parity'
        #     if self.verbose >= 1.:
        #         print(self.name + ": set conserve to", conserve)
        site = SpinHalfSite(conserve=None)
        return site

    def init_terms(self, model_params):
        J = get_parameter(model_params, 'J', 1., self.name, True)
        g = get_parameter(model_params, 'g', 1., self.name, True)
        h = get_parameter(model_params, 'h', 1., self.name, True)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
            self.add_onsite(-h, u, 'Sigmax')

        for u1, u2, dx in self.lat.nearest_neighbors:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)
        # done


class IsingChain(IsingModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """
    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)



def example_TEBD_tf_ising_lightcone(L, g, h, tmax, dt, chi, order, verbose=True):
    print("finite TEBD, real time evolution")
    print("L={L:d}, g={g:.2f}, tmax={tmax:.2f}, dt={dt:.3f}".format(L=L, g=g, tmax=tmax, dt=dt))
    print(" Create Product State ")

    # model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None, verbose=verbose)
    # M = TFIChain(model_params)
    # product_state = ["up"] * M.lat.N_sites
    # psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)


    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', h=h)
    # M = IsingChain({L:L, g:g, h:h})
    M = IsingChain(model_params)

    # M = TFIChain({'L': L})
    p_state = ["up"] * L
    psi = MPS.from_product_state(M.lat.mps_sites(), p_state, bc=M.lat.bc_MPS)

    wf_dir_path = 'data_tebd_dt%e/1d_%s_g%.4f_h%.4f/L%d/wf_chi%d_%s/' % (dt, 'TFI', g, h, L, chi, order)
    if not os.path.exists(wf_dir_path):
        os.makedirs(wf_dir_path)

    dt_measure = np.amax([0.05, dt])
    # tebd.Engine makes 'N_steps' steps of `dt` at once; for second order this is more efficient.
    order_int = int(order[0])
    tebd_params = {
        'order': order_int,
        'dt': dt,
        'N_steps': int(dt_measure // dt),
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'verbose': verbose,
    }
    eng = tebd.Engine(psi, M, tebd_params)
    S = [psi.entanglement_entropy()]
    Sz = [psi.expectation_value('Sz')]
    t_list = [0.]
    for n in range(int(tmax / dt_measure + 0.5)):
        eng.run()
        S.append(psi.entanglement_entropy())
        Sz.append(psi.expectation_value('Sz'))

        t_list.append((n+1)*dt_measure)
        time = (n+1)*dt_measure
        if np.isclose(time % 0.5, 0):
            nd_MPS = [t.to_ndarray() for t in psi._B]
            nd_MPS = mps_func.plr_2_lpr(nd_MPS)
            pickle.dump(nd_MPS, open(wf_dir_path + 'T%.1f.pkl' % t_list[-1],'wb'))

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(S[::-1],
    #            vmin=0.,
    #            aspect='auto',
    #            interpolation='nearest',
    #            extent=(0, L - 1., -0.5 * dt_measure, eng.evolved_time + 0.5 * dt_measure))
    # plt.xlabel('site $i$')
    # plt.ylabel('time $t/J$')
    # plt.ylim(0., tmax)
    # plt.colorbar().set_label('entropy $S$')
    # filename = 'c_tebd_lightcone_{g:.2f}.pdf'.format(g=g)
    # # plt.savefig(filename)
    # # print("saved " + filename)
    # plt.show()
    # # plt.plot(np.array(Sz)[:,L//2])
    # plt.imshow(np.array(Sz))
    # plt.show()

    dir_path = 'data_tebd_dt%e/1d_%s_g%.4f_h%.4f/L%d/' % (dt, 'TFI', g, h, L)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'mps_chi%d_%s_dt.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(t_list))

    filename = 'mps_chi%d_%s_sz_array.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(Sz)*2)

    filename = 'mps_chi%d_%s_ent_array.npy' % (chi, order)
    path = dir_path + filename
    np.save(path, np.array(S))


if __name__ == "__main__":
    for dt in [0.5, 0.1]:
        for h in [0., 0.1, 0.5, 0.9045]:
            for order in ['1st', '2nd', '4th']:
                example_TEBD_tf_ising_lightcone(L=31, g=1.4, h=h, tmax=5, dt=dt, chi=1024, order=order)
