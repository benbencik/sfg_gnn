import networkx
import torch_geometric
import matplotlib
import numpy
import torch 

class Generator():
    def __init__(self, model:str) -> None:
        self.model = model

    def uniform_spin_generation(self, n):
        # generate random spin as 3d unit vector
        RAD = 1.
        Theta = numpy.random.uniform(0., numpy.pi, n)  # random numbers between 0 and pi
        Phi = numpy.random.uniform(0., 2.*numpy.pi, n) # random numbers between 0 and 2*pi

        X = RAD*numpy.cos(Phi)*numpy.sin(Theta)
        Y = RAD*numpy.sin(Phi)*numpy.sin(Theta)
        Z = RAD*numpy.cos(Theta)
        
        spins = numpy.dstack([X, Y, Z], )[0]
        return spins


    def lattice_graph(self, width_nodes: int, height_nodes: int) -> torch_geometric.data.Data:
        lattice = networkx.grid_2d_graph(width_nodes, height_nodes, periodic=True)

        # networkx.draw(lattice)
        # matplotlib.pyplot.show()
        if self.model == "heisenberg": spins = self.uniform_spin_generation(len(lattice.nodes))
        elif self.model == "ising": spins = numpy.random.choice([-1, 1], size=(len(lattice.nodes), 1))
        for idx, node in enumerate(lattice.nodes()):
            lattice.nodes[node]['x'] = spins[idx]
            
        for edge in lattice.edges():
            weight = (numpy.random.random() * 2) - 1
            # weight = numpy.ones((len(edge_index[0])))
            lattice.edges[edge]['edge_attr'] = weight

        energy = 0
        for n1 in lattice.nodes():
            for n2 in lattice.neighbors(n1):
                if (n1, n2) in lattice.edges: e_weight = lattice.edges[n1, n2]['edge_attr']
                else: e_weight = lattice.edges[n2, n1]['edge_attr']
                n1_spin = lattice.nodes[n1]['x']
                n2_spin = lattice.nodes[n2]['x']
                if self.model == "heisenberg": energy += float(numpy.dot(n1_spin, n2_spin) * e_weight)
                elif self.model == "ising": energy += float(n1_spin * n2_spin * e_weight)
        
        data = torch_geometric.utils.from_networkx(lattice)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = energy
        return data
