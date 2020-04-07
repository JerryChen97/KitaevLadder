import matplotlib.pyplot as plt
from tenpy.models.lattice import Ladder

ax = plt.gca()
lat = Ladder(5, None, bc='periodic')
lat.plot_coupling(ax, lat.pairs['nearest_neighbors'], linewidth=3.)
lat.plot_order(ax=ax, linestyle='--')
lat.plot_sites(ax)
lat.plot_basis(ax, color='g', linewidth=2.)
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
