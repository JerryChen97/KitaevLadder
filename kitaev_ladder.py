import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt


from tenpy.networks.site import Site, SpinHalfFermionSite, SpinHalfSite, GroupedSite, SpinSite
from tenpy.tools.misc import to_iterable, to_iterable_of_len, inverse_permutation
from tenpy.networks.mps import MPS  # only to check boundary conditions

from tenpy.models.lattice import Lattice, _parse_sites
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import get_parameter

from tenpy.algorithms import dmrg
# from tenpy.networks import SpinHalfSite

__all__ = ['KitaevLadder']


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
        NNz = [(0, 1, np.array([0])), (2, 3, np.array([0]))]
        NNx = [(1, 3, np.array([0])), (2, 0, np.array([1]))]
        NNy = [(0, 2, np.array([0])), (3, 1, np.array([1]))]
        # nNN = [(0, 1, np.array([1])), (1, 0, np.array([1]))]
        # nnNN = [(0, 0, np.array([2])), (1, 1, np.array([2]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors_x', NNx)
        kwargs['pairs'].setdefault('nearest_neighbors_y', NNy)
        kwargs['pairs'].setdefault('nearest_neighbors_z', NNz)
        # kwargs['pairs'].setdefault('nearest_neighbors', NNx+NNy+NNz)
        # kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        # kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        Lattice.__init__(self, [L], sites, **kwargs)
        
class KitaevLadderModel(CouplingMPOModel):
    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', None, self.name)
        fs = SpinHalfSite(conserve=conserve)
        # gs = GroupedSite([fs, fs, fs, fs])
        return [fs, fs, fs, fs]

    def init_lattice(self, model_params):
        L = get_parameter(model_params, 'L', 3, self.name)
        gs = self.init_sites(model_params)
        lat = KitaevLadder(L, gs)
        return lat

    def init_terms(self, model_params):
        Jx = get_parameter(model_params, 'Jx', 1., self.name, True)
        Jy = get_parameter(model_params, 'Jy', 1., self.name, True)
        Jz = get_parameter(model_params, 'Jz', 1., self.name, True)

        for u1, u2, dx in self.lat.pairs['nearest_neighbors_z']:
            self.add_coupling(-Jz, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors_x']:
            self.add_coupling(-Jx, u1, 'Sigmax', u2, 'Sigmax', dx)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors_y']:
            self.add_coupling(-Jy, u1, 'Sigmay', u2, 'Sigmay', dx)

    
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


def run():

    data = dict(QL=[], ent_spectrum=[])

    model_params = dict(conserve=None, Jx=1., Jy=1., Jz=1., L=2, verbose=1)

    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'mixer_params': {
            'amplitude': 1.e-5,
            'decay': 1.2,
            'disable_after': 30
        },
        'trunc_params': {
            'svd_min': 1.e-10,
        },
        'lanczos_params': {
            'N_min': 5,
            'N_max': 20
        },
        'chi_list': {
            0: 9,
            10: 49,
            20: 100
        },
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'max_sweeps': 150,
        'verbose': 1.,
    }

    prod_state = ["up"] * (4 * model_params['L'])

    eng = None


    print("=" * 100)


    if eng is None:  # first time in the loop
        M = KitaevLadderModel(model_params)
        psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
        eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    else:
        del eng.engine_params['chi_list']
        M = KitaevLadderModel(model_params)
        eng.init_env(model=M)

    E, psi = eng.run()

    data['ent_spectrum'].append(psi.entanglement_spectrum(by_charge=True)[0])

    return data


def plot_results(data):

    plt.figure()
    ax = plt.gca()
    ax.plot(data['ent_spectrum'], marker='o')
    ax.set_xlabel(r"DMRG Step")
    ax.set_ylabel(r"$ E $")
    plt.savefig("KitaevLadderModel.pdf")

    # plt.figure()
    # ax = plt.gca()
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # color_by_charge = {}
    # for phi_ext, spectrum in zip(data['phi_ext'], data['ent_spectrum']):
    #     for q, s in spectrum:
    #         q = q[0]
    #         label = ""
    #         if q not in color_by_charge:
    #             label = "{q:d}".format(q=q)
    #             color_by_charge[q] = colors[len(color_by_charge) % len(colors)]
    #         color = color_by_charge[q]
    #         ax.plot(phi_ext * np.ones(s.shape),
    #                 s,
    #                 linestyle='',
    #                 marker='_',
    #                 color=color,
    #                 label=label)
    # ax.set_xlabel(r"$\Phi_y / 2 \pi$")
    # ax.set_ylabel(r"$ \epsilon_\alpha $")
    # ax.set_ylim(0., 8.)
    # ax.legend(loc='upper right')
    # plt.savefig("haldane_C3_ent_spec_flow.pdf")
    
# plot_lattice()

data = run()
# plot_results(data)
print(data)
