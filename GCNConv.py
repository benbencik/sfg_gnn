

import math
import wandb
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, GravNetConv, GATv2Conv

""" Create dataste of Pytorch Geometric Data objects """

nnodes = 150
nnfeature = 1
nefeatures = 1
dataset = []

for _ in range(1500):
    # undirected graph contains edges in both direcitons
    source_nodes, target_nodes = [], []
    edge_weights = []
    for i in range(1, nnodes+1):
        source_nodes += [(i-1) % nnodes, i % nnodes]
        target_nodes += [i % nnodes, (i-1) % nnodes]
        weight = random.randrange(10)
        edge_weights += [weight, weight]

    x = torch.tensor([[random.choice([-1, 1])] for _ in range(nnodes)], dtype=torch.float)
    edge_index = torch.LongTensor([source_nodes, target_nodes])
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # calculate resulting energy of the lattice
    energy = 0
    for i in range(1, nnodes):
        energy += x[i-1][0] * x[i][0] * edge_weights[(i-1)*2]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=energy/(nnodes*5))
    dataset.append(data)

# check if edges are encoded correctly
for data in dataset:
    assert data.edge_index.max() < data.num_nodes

""" Splitting and batching the dataset """
from torch_geometric.loader import DataLoader

split = (len(dataset) // 10) * 2
train_dataset = dataset[split:]
test_dataset = dataset[:split]

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)
print(f"Number of training graphs: {len(train_loader)}")
print(f"Number of testing graphs: {len(test_loader)}")


class GraphNetRegression(torch.nn.Module):
    def __init__(self, num_node_features, num_output_features):
        super(GraphNetRegression, self).__init__()
        self.conv1 = GCNConv(1, 32, improved=True)
        self.conv2 = GCNConv(32, 16, improved=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_weighs):
        x = self.conv1(x, edge_index, edge_weighs)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weighs)
        x = F.relu(x)
        
        x = self.fc(x)
        return x


num_node_features = 1
num_output_features = 1
model = GraphNetRegression(num_node_features, num_output_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model), len(dataset)

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
        # print(out)
        out = torch.sum(out)
        loss = F.mse_loss(out, data.y[0])  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader, epoch):
     model.eval()
     mse_loss = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_weight)  
        pred = torch.sum(out)
        mse_loss += (pred - data.y)**2
     return math.sqrt(float(mse_loss) / len(loader.dataset))

train_loss, test_loss = [], []
epoch = 20
for epoch in range(epoch):
    train()
    train_loss.append(test(train_loader, epoch))
    test_loss.append(test(test_loader, epoch))
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}')

plt.plot(range(epoch+1), train_loss, label="train")
plt.plot(range(epoch+1), test_loss, label="test")
plt.xlabel("Number of epochs")
plt.xticks(np.linspace(0, epoch, num=epoch//2, dtype=int))
plt.ylabel("RMSE loss")
plt.legend()
plt.show()


