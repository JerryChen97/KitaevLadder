import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt
from random import choice
import scipy.sparse as sparse

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

# functools
from functools import wraps

# path
from pathlib import Path

__all__ = ['KitaevLadder', 'KitaevLadderModel']


class KitaevLadderSnakeCompact(Lattice):    
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
        basis = np.array([[4., 0.]])
        pos = np.array([[0., 0.], [1., 0.], [3., 0.], [4., 0.]])
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        
        NNx = [(0, 2, np.array([0])), (3, 1, np.array([1]))]
        NNz = [(0, 1, np.array([0])), (2, 3, np.array([0]))]
        NNy = [(1, 3, np.array([0])), (0, 2, np.array([1]))]
#         nNNa = [(1, 2, np.array([0])), (3, 0, np.array([1]))]
#         nNNb = [(0, 3, np.array([0])), (2, 1, np.array([1]))]
        
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors_x', NNx)
        kwargs['pairs'].setdefault('nearest_neighbors_y', NNy)
        kwargs['pairs'].setdefault('nearest_neighbors_z', NNz)
#         kwargs['pairs'].setdefault('next_nearest_neighbors_a', nNNa)
#         kwargs['pairs'].setdefault('next_nearest_neighbors_b', nNNb)

        kwargs.setdefault('bc', 'open')
        
        Lattice.__init__(self, [L], sites, **kwargs)
        
        
class KitaevLadderSnakeCompactModel(CouplingMPOModel):
    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        # conserve = get_parameter(model_params, 'conserve', None, self.name)
        conserve = model_params.get('conserve', None)
        S = model_params.get('S', 0.5)
        if S==0.5:
            fs = SpinHalfSite(conserve=conserve)
        else:
            fs = SpinSite(S=S, conserve=conserve)
        return [fs, fs, fs, fs]

    def init_lattice(self, model_params):
        L = model_params.get('L', 4) # 2 is the least possible number for L to be a Kitaev ladder we want, and 4 is more secured (I hope so)

        gs = self.init_sites(model_params)
        model_params.pop("L")


        order = model_params.get('order', 'default')
        bc = model_params.get('bc', 'open')
        bc_MPS=model_params.get('bc_MPS', 'finite')
        lattice_params = dict(
            order=order,
            bc=bc,
            bc_MPS=bc_MPS,
            basis=None,
            positions=None,
            nearest_neighbors=None,
            next_nearest_neighbors=None,
            next_next_nearest_neighbors=None,
            pairs={},
        )

        lat = KitaevLadderSnakeCompact(L, gs, **lattice_params)
        return lat

    def init_terms(self, model_params):
        # Jx = get_parameter(model_params, 'Jx', 1., self.name, True)
        # Jy = get_parameter(model_params, 'Jy', 1., self.name, True)
        # Jz = get_parameter(model_params, 'Jz', 1., self.name, True)
        Jx = model_params.get('Jx', 1.)
        Jy = model_params.get('Jy', 1.)
        Jz = model_params.get('Jz', 1.)


        S = model_params.get('S', 0.5)
        if S==0.5:
            for u1, u2, dx in self.lat.pairs['nearest_neighbors_x']:
                self.add_coupling(Jx, u1, 'Sigmax', u2, 'Sigmax', dx)
            for u1, u2, dx in self.lat.pairs['nearest_neighbors_y']:
                self.add_coupling(Jy, u1, 'Sigmay', u2, 'Sigmay', dx)         
            for u1, u2, dx in self.lat.pairs['nearest_neighbors_z']:
                self.add_coupling(Jz, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        else:
            for u1, u2, dx in self.lat.pairs['nearest_neighbors_x']:
                self.add_coupling(Jx, u1, 'Sx', u2, 'Sx', dx)
            for u1, u2, dx in self.lat.pairs['nearest_neighbors_y']:
                self.add_coupling(Jy, u1, 'Sy', u2, 'Sy', dx)         
            for u1, u2, dx in self.lat.pairs['nearest_neighbors_z']:
                self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
        
def plot_lattice():
    fig, ax = plt.subplots()
    lat = KitaevLadderSnakeCompact(5, None, bc='open')
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

def run_atomic(
    # model parameters
    chi=30, 
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=4, 
    S=.5, 
    bc='periodic', 
    bc_MPS='infinite', 
    # dmrg parametersc
    initial_psi=None, # input psi
    initial='random', 
    max_E_err=1.e-6, 
    max_S_err=1.e-4, 
    max_sweeps=200, 
    N_sweeps_check=10, 
    canonicalized=True, 
    # control for the verbose output
    verbose=1, 
):
    """ 
        The fundamental function for running DMRG
    """

    #######################
    # set the paramters for model initialization
    model_params = dict(
        conserve=None, 
        Jx=Jx, 
        Jy=Jy, 
        Jz=Jz, 
        L=L, 
        S=S,
        verbose=verbose,
        bc=bc,
        bc_MPS=bc_MPS,
        )
    # initialize the model
    M = KitaevLadderSnakeCompactModel(model_params)
    # providing a product state as the initial state
    # prod_state = ["up", "up"] * (2 * model_params['L'])
    # random generated initial state
    if initial_psi==None:
        prod_state = [ choice(["up", "down"]) for i in range(4 * L)]
        if initial == 'up':
            prod_state = ["up" for i in range(4 * L)]
        if initial == 'down':
            prod_state = ["down" for i in range(4 * L)]
        psi = MPS.from_product_state(
            M.lat.mps_sites(), 
            prod_state, 
            bc=M.lat.bc_MPS,
        )
    else:
        psi = initial_psi.copy()
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
        'max_E_err': max_E_err,
        'max_S_err': max_S_err,
        'max_sweeps': max_sweeps,
        'N_sweeps_check': N_sweeps_check,
        'verbose': verbose,
    }
    #######################
    
    if verbose:
        print("\n")
        print("=" * 80)
        print("="*30 + "START" + "="*30)
        print("=" * 80)
        print("Chi = ", chi, '\n')

    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    eng.reset_stats()
    eng.trunc_params['chi_max'] = chi
    info = eng.run()

    if canonicalized:
        psi.canonical_form()
        if verbose:
            print("Before the canonicalization:")
            print("Bond dim = ", psi.chi)

            print("Canonicalizing...")
            psi_before = psi.copy()


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
        
    # the wave function, the ground-state energy, and the DMRG engine are all that we need
    result = dict(
        psi=psi.copy(),
        energy=energy,
        sweeps_stat=eng.sweep_stats.copy(),
        parameters=dict(
            # model parameters
            chi=chi,
            Jx=Jx, 
            Jy=Jy, 
            Jz=Jz, 
            L=L, 
            # dmrg parameters
            initial_psi=initial_psi,
            initial=initial,
            max_E_err=max_E_err,
            max_S_err=max_S_err,
            max_sweeps=max_sweeps,
        )
    )
    return result

def naming(
    # model parameters
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=4, 
    ):
    return "KitaevLadder"+"_chi_"+str(chi)+"_Jx_"+str(Jx)+"_Jy_"+str(Jy)+"_Jz_"+str(Jz)+"_L_"+str(L)

def full_path(
    # model parameters
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=4, 
    prefix='data/', suffix='.h5',
    **kwargs):
    return prefix+naming(chi, Jx, Jy, Jz, L)+suffix
    
def save_after_run(run, folder_prefix='data/'):
    """
        Save data as .h5 files
    """
    @wraps(run)
    def wrapper(*args, **kwargs):
        # if there is no such folder, create another one; if exists, doesn't matter
        Path(folder_prefix).mkdir(parents=True, exist_ok=True)
        file_name = full_path(prefix=folder_prefix, **kwargs)
        
        # if the file already existed then don't do the computation again
        if os.path.isfile(file_name):
            print("This file already existed. Pass.")
            return 0
        else:
            result = run(*args, **kwargs)
            with h5py.File(file_name, 'w') as f:
                hdf5_io.save_to_hdf5(f, result)
                
            return result
    
    return wrapper

def load_data(
    chi=30,
    Jx=1., 
    Jy=1., 
    Jz=0., 
    L=4, 
    prefix='data/', 
):
    file_name = full_path(chi, Jx, Jy, Jz, L, prefix=prefix, suffix='.h5')
    if not Path(file_name).exists():
        return -1
    with h5py.File(file_name, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        return data

def finite_scaling(
    # model params, should be input
    Jx = 0.5,
    Jy = 0.5,
    Jz = 0,
    L = 4,
    S=.5,
    

    # next there are some DMRG params
    # tolerance for entropy calc error, should be input
    max_S_err = 1e-4,
    N_sweeps_check = 5,
    max_sweeps = 1000,

    # bond dimension list, should be input
    chi_list = range(8, 50, 2),
    
    # initial wave function
    psi = None,

    verbose = 1,
    
    # should we load the existing files and also save the results into files
    save_and_load = False,

    prefix = 'snake/',
    
    # should we do plotting after calculation
    plot = False,
):
    """
        Computing the finite-scaling cases at a specific `J=(Jx, Jy, Jz)`, over a specific bond dimension region.
    """
    
    if save_and_load:
        # folder name
        # prefix = f'data_L_{L}_comb/'
        # if there is no such folder, create another one; if exists, doesn't matter
        Path(prefix).mkdir(parents=True, exist_ok=True)
        run_save = save_after_run(run_atomic, folder_prefix=prefix)


    S_list = []
    xi_list = []

    psi = psi
    
    for chi in chi_list:
        
        if save_and_load:
            data = run_save(
                Jx = Jx,
                Jy = Jy,
                Jz = Jz,
                L = L,
                S= S,
                max_S_err=max_S_err,
                chi = chi,
                initial_psi=psi,
                N_sweeps_check=N_sweeps_check,    
                max_sweeps=max_sweeps, 
                verbose=verbose,
            )

            if data==0:
                data = load_data(Jx = Jx, Jy = Jy, Jz = Jz, L = L, chi = chi, prefix = prefix)
                pass
            pass
        else:
            data = run_atomic(
                Jx = Jx,
                Jy = Jy,
                Jz = Jz,
                L = L,
                S = S,
                max_S_err=max_S_err,
                chi = chi,
                initial_psi=psi,
                N_sweeps_check=N_sweeps_check,    
                max_sweeps=max_sweeps, 
                verbose=verbose,
            )
            pass
        
        psi = data['psi']
        S_list.append(np.mean(psi.entanglement_entropy()))
        xi_list.append(psi.correlation_length())        
        pass
    
    if plot:
        start = 0
        end = -1

        xs = np.log(xi_list[start:end])
        ys = S_list[start:end]

        def func(log_xi, c, a):
            return (c / 6) * log_xi + a
        fitParams, fitCovariances = curve_fit(func, xs, ys)

        plt.plot(xs, ys, 'o', label='Data Points')
        plt.xlabel(r'Log of Correlation Length, $\log\xi$')
        plt.ylabel(r'Average Entanglement Entropy, $S$')

        fitting_label = r'Curve Fitting: $S = \frac{c}{6}\log\xi + b$, c = %.2f, b= %.2f' % (fitParams[0], fitParams[1])
        plt.plot(xs, func(xs, fitParams[0], fitParams[1]), label=fitting_label)

        plt.legend()
        title_name = f'Finite Scaling at J = ({Jx}, {Jy}, {Jz}), L={L}'
        plt.title(title_name)
        plt.savefig(title_name + '.png')

        plt.show()
        pass
    return S_list, xi_list

def fDMRG_KL(
    Jx=1., 
    Jy=1., 
    Jz=1., 
    L=4, 
    chi=100, 
    verbose=True, 
    order='default', 
    bc_MPS='finite', 
    bc='open',
    # to extract the low-lying excitation
    orthogonal_to={}, 
    ):
    """
        The finite DMRG for Kitaev Ladders
    """
    
    print("finite DMRG, Kitaev ladder model")
    print("L = {L:d}, Jx = {Jx:.2f}, Jy = {Jy:.2f}, Jz = {Jz:.2f}, ".format(L=L, Jx=Jx, Jy=Jy, Jz=Jz))
    model_params = dict(L=L, Jx=Jx, Jy=Jy, Jz=Jz, bc_MPS=bc_MPS, bc=bc, conserve=None, order=order, verbose=verbose)
    M = KitaevLadderSnakeCompactModel(model_params)
    
    print("bc_MPS = ", M.lat.bc_MPS)
    
    product_state = [np.random.choice(["up", "down"]) for i in range(M.lat.N_sites)]
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
#         'mixer': None,  # setting this to True helps to escape local minima
        'mixer': True,
        'mixer_params': {
            'amplitude': 1.e-4,
            'decay': 1.2,
            'disable_after': 50
        },
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10
        },
        'verbose': verbose,
        'combine': True,
        'orthogonal_to': orthogonal_to,
    }
    info = dmrg.run(psi, M, dmrg_params)  # the main work...
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi) 

    return E, psi, M