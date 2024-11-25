import torch

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import GATConv

import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self):
        """
        We first define and stack three graph convolution layers,
        which corresponds to aggregating 3-hop neighborhood information around each node
        (all nodes up to 3 "hops" away)
        """
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(2, 2) #[Number of features, dimension reduction]
        self.conv2 = GCNConv(2, 2)
        self.conv3 = GCNConv(2, 2)
        self.classifier = Linear(2, 1) #[Last dimension, 1 for binary classification]

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = h.tanh()
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = h.tanh()
        h = self.conv3(h, edge_index, edge_weight=edge_weight)
        h = h.tanh()  # Final GNN embedding space.

        # Apply classifier
        logits = self.classifier(h)

        # We can apply a sigmoid to the output or just leave it like this 
        # and then use sigmoid in the loss function
        out = logits

        return out, h

class GAT(torch.nn.Module):
    """
    In the Graph Attetion Network (GAT), every node updates its features vector using the
    importance of the features of the neighbouring nodes (including itself).
    PyTorchGeometric GATConv allow us to multihead, i.e., repeat the attention matrix calculation
    as many times as we want.

    Here, we use two GATConv layers in which we update the nodes feature vectors. In the forward
    phase we use dropout renormalization and save the output as it is since we're going to use
    a BCEWithLogitsLoss as loss function.
    """
    def __init__(self, num_features, hidden_channels, dropout_gat, dropout_fc):
        super(GAT, self).__init__()
        self.hid          = hidden_channels
        self.in_head      = 8
        self.out_head     = 1
        self.dropout_gat  = dropout_gat
        self.dropout_fc   = dropout_fc

        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head, dropout=self.dropout_gat)
        self.conv2 = GATConv(self.hid * self.in_head, 1, concat=False, heads=self.out_head, dropout=self.dropout_fc) # Binary Classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout_gat, training=self.training)
        h = self.conv1(x, edge_index)
        h = F.elu(h)

        h = F.dropout(h, p=self.dropout_gat, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.elu(h)

        out = h

        return out.squeeze(1)
