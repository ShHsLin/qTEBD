"""Example illustrating the use of DMRG in tenpy.

The example functions in this class do the same as the ones in `toycodes/d_dmrg.py`,
but make use of the classes defined in tenpy.
"""
# Copyright 2018 TeNPy Developers

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg

import os, sys
sys.path.append('..')
import misc
import parse_args



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







def example_DMRG_tf_ising_finite(L, g, h=0, chi=2, verbose=True):
    print("finite DMRG, transverse field Ising model")
    print("L={L:d}, g={g:.2f}, h={h:.2f}".format(L=L, g=g, h=h))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', h=h)
    M = IsingChain(model_params)

    # model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None, verbose=verbose)
    # M = TFIChain(model_params)

    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-16,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-12
        },
        'verbose': verbose,
    }
    info = dmrg.run(psi, M, dmrg_params)  # the main work...
    E = info['E']
    print("E = {E:.16f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.sum(psi.expectation_value("Sigmax"))
    mag_z = np.sum(psi.expectation_value("Sigmaz"))
    print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    return E, psi, M


def example_DMRG_tf_ising_infinite(g, verbose=True):
    print("infinite DMRG, transverse field Ising model")
    print("g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None, verbose=verbose)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
        'verbose': verbose,
    }
    # instead of
    #  info = dmrg.run(psi, M, dmrg_params)
    #  E = info['E']
    # we can also use the a Engine directly:
    eng = dmrg.EngineCombine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.mean(psi.expectation_value("Sigmax"))
    mag_z = np.mean(psi.expectation_value("Sigmaz"))
    print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
    print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
    print("correlation length:", psi.correlation_length())
    # compare to exact result
    from tfi_exact import infinite_gs_energy
    E_exact = infinite_gs_energy(1., g)
    print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
    print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M

def example_DMRG_heisenberg_xxz_finite(L, Jz, chi, conserve='best', verbose=True):
    print("finite DMRG, Heisenberg XXZ chain")
    print("L={L:d}, Jz={Jz:.2f}".format(L=L, Jz=Jz))
    model_params = dict(L=L, S=0.5, Jx=1., Jy=1., Jz=Jz,
                        bc_MPS='finite', conserve=conserve, verbose=verbose)
    M = SpinModel(model_params)
    product_state = ["up", "down"] * (M.lat.N_sites // 2)
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-16,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-16
        },
        'verbose': verbose,
    }
    info = dmrg.run(psi, M, dmrg_params)  # the main work...
    E = info['E']
    E = E * 4
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    mag_z = np.mean(Sz)
    print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],
                                                                     Sz1=Sz[1],
                                                                     mag_z=mag_z))
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
    print("correlations <Sz_i Sz_j> =")
    print(corrs)
    return E, psi, M

def example_DMRG_heisenberg_xxz_infinite(Jz, conserve='best', verbose=True):
    print("infinite DMRG, Heisenberg XXZ chain")
    print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Jz, conserve=conserve))
    model_params = dict(
        L=2,
        S=0.5,  # spin 1/2
        Jx=1.,
        Jy=1.,
        Jz=Jz,  # couplings
        bc_MPS='infinite',
        conserve=conserve,
        verbose=verbose)
    M = SpinModel(model_params)
    product_state = ["up", "down"]  # initial Neel state
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-10,
        'verbose': verbose,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    mag_z = np.mean(Sz)
    print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],
                                                                     Sz1=Sz[1],
                                                                     mag_z=mag_z))
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    print("correlation length:", psi.correlation_length())
    corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
    print("correlations <Sz_i Sz_j> =")
    print(corrs)
    return E, psi, M


if __name__ == "__main__":
    args = parse_args.parse_args()

    L = args.L
    Hamiltonian = args.H
    g = args.g
    h = args.h
    chi = args.chi
    assert Hamiltonian in ['TFI', 'XXZ']

    if Hamiltonian == 'TFI':
        dmrg_E, psi, M = example_DMRG_tf_ising_finite(L=L, g=g, h=h, chi=chi)
    else:
        dmrg_E, psi, M = example_DMRG_heisenberg_xxz_finite(L=L, Jz=g, chi=chi)

    # print("-" * 100)
    # example_DMRG_tf_ising_infinite(g=1.5)
    # print("-" * 100)
    # example_DMRG_heisenberg_xxz_infinite(Jz=1.5)

    dir_path = 'data/1d_%s_g%.1f_h%.1f/' % (Hamiltonian, g, h)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'dmrg_chi%d_energy.csv' % chi
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
        E_dict[L] = dmrg_E
        misc.save_array(path, misc.dict_2_nparray(E_dict))
        # If no data --> generate data
        print("Save new data")

