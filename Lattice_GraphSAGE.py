import pandas
import seaborn
import numpy as numpy
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch_geometric
import torch_geometric.nn.models
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from torch.utils.tensorboard import SummaryWriter

import graph_generator


""" 
Splitting and batching the dataset 

A data loader which merges data objects from a 
torch_geometric.data.Dataset to a mini-batch. 
Data objects can be either of type Data or HeteroData.
"""
model = "ising"
gg = graph_generator.Generator(model=model)
dataset = [gg.lattice_graph(5,20) for _ in range(1000)]
print(dataset[0])

split = (len(dataset) // 10) * 2
train_dataset = dataset[split:]
test_dataset = dataset[:split]

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=10)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=5)
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

if model == "ising": in_features = 1
elif model == "heisenberg": in_features = 3
out_features = 1

model = torch_geometric.nn.models.GraphSAGE(
    in_channels=in_features, 
    out_channels=out_features,
    hidden_channels=256,
    num_layers=4,
    jk='max'
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

""" Model parameters """

print("Sum of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


def train():
    model.train()
    # Iterate in batches over the training dataset.
    for data in train_loader:
        # Perform a single forward pass.
        out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        out = torch_geometric.nn.pool.global_mean_pool(out, data.batch)
        
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
            out = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            out = torch_geometric.nn.pool.global_mean_pool(out, data.batch)
            loss = F.mse_loss(out.squeeze(), data.y.squeeze()) 
            pred_nums = [str(num) for num in out.squeeze().tolist()]
            true_nums = [str(num) for num in data.y.squeeze().tolist()]
            f.write(f"pred: {', '.join(pred_nums)}\n")
            f.write(f"true: {', '.join(true_nums)}\n")
            f.write("-------------\n")
            mse_loss += loss
     return float(mse_loss) / len(loader)

epoch = 50
writer = SummaryWriter()
train_loss, test_loss = [], []

for epoch in range(epoch+1):
    train()
    train_loss.append(test(train_loader))
    test_loss.append(test(test_loader))
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss[-1]:.5f}, Test Loss: {test_loss[-1]:.5f}')
    writer.add_scalar('Loss/train', train_loss[-1], epoch)

writer.flush()
loss_df = pandas.DataFrame({'epochs': range(epoch+1), 'train_loss': train_loss, 'test_loss': test_loss})

seaborn.set_theme()
loss_plot = seaborn.lineplot(loss_df[['train_loss', 'test_loss']])
loss_plot.set_title("MSE")
loss_plot.set_xlabel("# Epochs")
loss_plot.set_ylabel("MSE loss")
plt.show()


