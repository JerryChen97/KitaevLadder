from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools import get_parameter
from tenpy.networks import SpinHalfSite

__all__ = ['KitaevLadder']


class KitaevLadder(CouplingMPOModel):
    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        conserve = get_parameter(model_params, 'conserve', 'parity', self.name)
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = SpinHalfSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        J = get_parameter(model_params, 'J', 1., self.name, True)
        g = get_parameter(model_params, 'g', 1., self.name, True)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)
        # done
