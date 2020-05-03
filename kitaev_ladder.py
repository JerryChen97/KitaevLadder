import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt
from random import choice

import tenpy
from tenpy.networks.site import Site, SpinHalfFermionSite, SpinHalfSite, GroupedSite, SpinSite
from tenpy.tools.misc import to_iterable, to_iterable_of_len, inverse_permutation
from tenpy.networks.mps import MPS  # only to check boundary conditions

from tenpy.models.lattice import Lattice, _parse_sites
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import get_parameter

from tenpy.algorithms import dmrg
# from tenpy.networks import SpinHalfSite

# some api for the file operation
import h5py
from tenpy.tools import hdf5_io
import os.path
# data path
prefix = 'data/'

__all__ = ['KitaevLadder', 'KitaevLadderModel']


class KitaevLadder(Lattice):    
    """ A ladder coupling two chains of the Kitaev form
    .. image :: /images/lattices/Ladder.*
    Parameters
    ----------
    L : int
        The length of each chain, we have 2*L sites in total.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both chains.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 1

    def __init__(self, L, sites, **kwargs):
        sites = _parse_sites(sites, 4)
        basis = np.array([[2., 0.]])
        pos = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        kwargs.setdefault('bc', 'periodic')
        kwargs.setdefault('bc_MPS', 'infinite')
        
        NNz = [(0, 1, np.array([0])), (2, 3, np.array([0]))]
        NNx = [(1, 3, np.array([0])), (2, 0, np.array([1]))]
        NNy = [(0, 2, np.array([0])), (3, 1, np.array([1]))]
        nNNa = [(1, 2, np.array([0])), (3, 0, np.array([1]))]
        nNNb = [(0, 3, np.array([0])), (2, 1, np.array([1]))]
        
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors_x', NNx)
        kwargs['pairs'].setdefault('nearest_neighbors_y', NNy)
        kwargs['pairs'].setdefault('nearest_neighbors_z', NNz)
        kwargs['pairs'].setdefault('next_nearest_neighbors_a', nNNa)
        kwargs['pairs'].setdefault('next_nearest_neighbors_b', nNNb)
        
        Lattice.__init__(self, [L], sites, **kwargs)
        
        
class KitaevLadderModel(CouplingMPOModel):
    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', None, self.name)
        fs = SpinHalfSite(conserve=conserve)
        return [fs, fs, fs, fs]

    def init_lattice(self, model_params):
        L = get_parameter(model_params, 'L', 3, self.name)
        gs = self.init_sites(model_params)
        model_params.pop("L")
        lat = KitaevLadder(L, gs)
        return lat

    def init_terms(self, model_params):
        Jx = get_parameter(model_params, 'Jx', 1., self.name, True)
        Jy = get_parameter(model_params, 'Jy', 1., self.name, True)
        Jz = get_parameter(model_params, 'Jz', 1., self.name, True)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors_z']:
            self.add_coupling(-Jz, u1, 'Sx', u2, 'Sx', dx)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors_x']:
            self.add_coupling(-Jx, u1, 'Sz', u2, 'Sz', dx)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors_y']:
            self.add_coupling(-Jy, u1, 'Sy', u2, 'Sy', dx)
         
        
def plot_lattice():
    fig, ax = plt.subplots()
    lat = KitaevLadder(5, None, bc='periodic')
    links_name = 'nearest_neighbors_z'
    lat.plot_coupling(ax, lat.pairs[links_name], linewidth=5.)
    # print(lat.pairs['nearest_neighbors'])
    print(lat.unit_cell)
    lat.plot_order(ax=ax, linestyle='--')
    lat.plot_sites(ax)
    # lat.plot_basis(ax, color='g', linewidth=3.)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    # plt.title(links_name)
    plt.show()

""" The fundamental function for running DMRG
"""
def run_atomic(
    # model parameters
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=1, 
    # dmrg parameters
    psi=None,
    initial='random',
    max_E_err=1.e-6,
    max_S_err=1.e-3,
    max_sweeps=10000,
    # control for the verbose output
    verbose=1, 
):

    #######################
    # set the paramters for model initialization
    model_params = dict(conserve=None, Jx=Jx, Jy=Jy, Jz=Jz, L=L, verbose=verbose)
    # initialize the model
    M = KitaevLadderModel(model_params)
    # providing a product state as the initial state
    # prod_state = ["up", "up"] * (2 * model_params['L'])
    # random generated initial state
    if psi==None:
        prod_state = [ choice(["up", "down"]) for i in range(4 * model_params['L'])]
        if initial == 'up':
            prod_state = ["up" for i in range(len(prod_state))]
        if initial == 'down':
            prod_state = ["down" for i in range(len(prod_state))]
        psi = MPS.from_product_state(
            M.lat.mps_sites(), 
            prod_state, 
            bc=M.lat.bc_MPS,
        )
    #######################

    
    #######################
    # set the parameters for the dmrg routine
    dmrg_params = {
        'start_env': 10,
#         'mixer': False,  # setting this to True helps to escape local minima
        'mixer': True,
        'mixer_params': {
            'amplitude': 1.e-4,
            'decay': 1.2,
            'disable_after': 50
        },
        'trunc_params': {
            'chi_max': 4,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-6,
        'max_S_err': 1.e-3,
        'max_sweeps': 10000,
        'verbose': verbose,
    }
    #######################
    
    if verbose:
        print("\n")
        print("=" * 80)
        print("="*30 + "START" + "="*30)
        print("=" * 80)
        print("Chi = ", chi, '\n')

    # here we create another new engine for every new chi
    # this is quite unusual but since some strange issues
    # we can only treat like this
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    eng.reset_stats()
    eng.trunc_params['chi_max'] = chi
    info = eng.run()
#         print("INFO: \n", info)

    if verbose:
        print("Before the canonicalization:")
        print("Bond dim = ", psi.chi)

        print("Canonicalizing...")
        psi_before = psi.copy()

    psi.canonical_form()

    if verbose:
        ov = psi.overlap(psi_before, charge_sector=0)
        print("The norm is: ",psi.norm)
        print("The overlap is: ", ov)
        print("After the canonicalization:")
        print("Bond dim = ", psi.chi)

        print("Computing properties")

    energy=info[0]

    if verbose:
        print("Optimizing")

    tenpy.tools.optimization.optimize(3)

    if verbose:
        print("Loop for chi=%d done." % chi)
        print("=" * 80)
        print("="*30 + " END " + "="*30)
        print("=" * 80)
        
    return (energy, psi.copy())

def naming(chi, Jx, Jy, Jz, L):
    return "KitaevLadder"+"_chi_"+str(chi)+"_Jx_"+str(Jx)+"_Jy_"+str(Jy)+"_Jz_"+str(Jz)+"_L_"+str(L)

def full_path(chi, Jx, Jy, Jz, L):
    return prefix+naming(chi, Jx, Jy, Jz, L)+".h5"
""" run the atomic and then save it
"""
def run_save(
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=1, 
    verbose=1, 
    calc_correlation=True,
    psi=None
):
    file_name = full_path(chi, Jx, Jy, Jz, L)
    
    # if the file already existed then don't do the computation again
    if os.path.isfile(file_name):
        print("This file already existed")
        psi = read_psi(chi=chi, Jx=Jx, Jy=Jy, Jz=Jz, L=L)
    else:
        (energy, psi) = run_atomic(
            chi=chi, 
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L, 
            verbose=verbose, 
            calc_correlation=calc_correlation, 
            psi=psi
        )
        data = {
            "psi": psi,
            "energy": energy,
            "parameters": {
                "chi": chi,
                "Jx": Jx,
                "Jy": Jy,
                "Jz": Jz,
                "L": L,
            }
        }
        with h5py.File(file_name, 'w') as f:
            hdf5_io.save_to_hdf5(f, data)
    return psi

def read_psi(
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=1, 
):
    file_name = full_path(chi, Jx, Jy, Jz, L)
    with h5py.File(file_name, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        # or for partial reading:
        psi = hdf5_io.load_from_hdf5(f, "/psi")
        return psi
    
def read_energy(
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=1, 
):
    file_name = full_path(chi, Jx, Jy, Jz, L)
    with h5py.File(file_name, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        # or for partial reading:
        energy = hdf5_io.load_from_hdf5(f, "/energy")
        return energy
    
def read_parameters(
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=1, 
):
    file_name = full_path(chi, Jx, Jy, Jz, L)
    with h5py.File(file_name, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        # or for partial reading:
        parameters = hdf5_io.load_from_hdf5(f, "/parameters")
        return parameters

# energy, psi = run_atomic(chi=100)
# print(psi.entanglement_entropy())
