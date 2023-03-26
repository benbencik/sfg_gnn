import networkx
import torch_geometric
import matplotlib
import numpy
import torch 

class GraphGenerator():
    def __init__(self) -> None:
        pass
    
    def lattice_graph(self, width_nodes: int, height_nodes: int) -> torch_geometric.data.Data:
        lattice = networkx.grid_2d_graph(width_nodes, height_nodes)

        # networkx.draw(lattice)
        # matplotlib.pyplot.show()

        # generate random spin as 3d unit vector
        for node in lattice.nodes():
            particle_spin = (numpy.random.random(size=(1, 3)) * 2) - 1
            normalized = particle_spin / numpy.linalg.norm(particle_spin)
            lattice.nodes[node]['x'] = normalized[0]
            
        for edge in lattice.edges():
            weight = (numpy.random.random(size=(1)) * 2) - 1
            lattice.edges[edge]['edge_attr'] = weight

        energy = 0
        for n1 in lattice.nodes():
            for n2 in lattice.neighbors(n1):
                if (n1, n2) in lattice.edges: e_weight = lattice.edges[n1, n2]['edge_attr']
                else: e_weight = lattice.edges[n2, n1]['edge_attr']
                n1_spin = lattice.nodes[n1]['x']
                n2_spin = lattice.nodes[n2]['x']
                energy += float(numpy.dot(n1_spin, n2_spin) * e_weight)
        
        data = torch_geometric.utils.from_networkx(lattice)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = energy
        return data

gg = GraphGenerator()
d = gg.lattice_graph(3,4)

# -----------------------------------------------------------------------

# %%
import math
import random
import pandas
import seaborn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.dense import Linear
from torch_geometric.nn import GCNConv, GraphConv, pool, GATConv
from torch_geometric.utils import to_networkx, add_remaining_self_loops
from torch_geometric.loader import DataLoader

from torch.utils.tensorboard import SummaryWriter

# %%
""" Create dataste of Pytorch Geometric Data objects """

nnodes = 50
dataset = []

for _ in range(2000):
    data = gg.lattice_graph(5,10)
    dataset.append(data)

# check if edges are encoded correctly
for data in dataset:
    assert data.edge_index.max() < data.num_nodes


""" 
Splitting and batching the dataset 

A data loader which merges data objects from a 
torch_geometric.data.Dataset to a mini-batch. 
Data objects can be either of type Data or HeteroData.
"""


split = (len(dataset) // 10) * 2
train_dataset = dataset[split:]
test_dataset = dataset[:split]


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=10)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=5)
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

# %%



# %%
""" Architecture of the neural network """

class GraphNetRegression(torch.nn.Module):
    def __init__(self, num_node_features, num_output_features):
        super(GraphNetRegression, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, add_self_loops=True)
        self.conv2 = GATConv(8, 256, add_self_loops=True, dropout=0.1)
        self.fc = nn.Linear(256, num_output_features)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
    
        x = self.fc(x)
        x = pool.global_add_pool(x, batch)
        return x


# %%
num_node_features = 3
num_output_features = 1
model = GraphNetRegression(num_node_features, num_output_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    # Iterate in batches over the training dataset.
    for data in train_loader:
        # Perform a single forward pass.
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        # print(out.shape, data.y.shape)
        loss = F.mse_loss(out.squeeze(), data.y) 
        
        # Derive gradients
        loss.backward()
        
        # Update parameters based on gradients.
        optimizer.step()
        optimizer.zero_grad()

def test(loader):
     model.eval()
     mse_loss = 0
     with open('prediciton' ,'w') as f:
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
            loss = F.mse_loss(out.squeeze(), data.y.squeeze()) 
            pred_nums = [str(num) for num in out.squeeze().tolist()]
            true_nums = [str(num) for num in data.y.squeeze().tolist()]
            f.write(f"pred: {', '.join(pred_nums)}\n")
            f.write(f"true: {', '.join(true_nums)}\n")
            f.write("-------------\n")
            mse_loss += loss
     return float(mse_loss) / len(loader)

# %%
""" Model parameters """

# print(model)
# print("Trainable parameters of the model:")
# for p in model.parameters():
#     if p.requires_grad: print('\t', p.shape)
# print("Sum of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

# %%
epoch = 20
# writer = SummaryWriter()
train_loss, test_loss = [], []

for epoch in range(epoch+1):
    train()
    train_loss.append(test(train_loader))
    test_loss.append(test(test_loader))
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss[-1]:.5f}, Test Loss: {test_loss[-1]:.5f}')
    # writer.add_scalar('Loss/train', train_loss[-1], epoch)

# writer.flush()
loss_df = pandas.DataFrame({'epochs': range(epoch+1), 'train_loss': train_loss, 'test_loss': test_loss})

seaborn.set_theme()
loss_plot = seaborn.lineplot(loss_df[['train_loss', 'test_loss']])
matplotlib.pyplot.show()
# loss_plot = seaborn.lineplot(test_loss)
loss_plot.set_title("MSE")
loss_plot.set_xlabel("# Epochs")
loss_plot.set_ylabel("MSE loss")



