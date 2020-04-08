import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt


from tenpy.networks.site import Site
from tenpy.tools.misc import to_iterable, to_iterable_of_len, inverse_permutation
from tenpy.networks.mps import MPS  # only to check boundary conditions

from tenpy.models.lattice import Lattice, _parse_sites
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
# from tenpy.tools import get_parameter
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
        kwargs['pairs'].setdefault('nearest_neighbors_z', NNz)
        kwargs['pairs'].setdefault('nearest_neighbors_x', NNx)
        kwargs['pairs'].setdefault('nearest_neighbors_y', NNy)
        kwargs['pairs'].setdefault('nearest_neighbors', NNx+NNy+NNz)
        # kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        # kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        Lattice.__init__(self, [L], sites, **kwargs)
        
fig, ax = plt.subplots()
lat = KitaevLadder(5, None, bc='periodic')
links_name = 'nearest_neighbors_z'
lat.plot_coupling(ax, lat.pairs[links_name], linewidth=5.)
# print(lat.pairs['nearest_neighbors'])
print(lat.unit_cell)
# lat.plot_order(ax=ax, linestyle='--')
lat.plot_sites(ax)
# lat.plot_basis(ax, color='g', linewidth=3.)
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
# plt.title(links_name)
plt.show()
