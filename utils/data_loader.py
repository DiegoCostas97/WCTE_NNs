# Some imports

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np
import networkx          as nx

import torch
import sys
import os

# Import Diego's tools
sys.path.append("/home/usc/ie/dcr/hk/ambe_analysis/paquetes")
from npz_to_df import digihits_info_to_df

from scipy.spatial        import KDTree
from torch_geometric.data import Data, Dataset

# Creation of the DataFrame the DataLoader is reading the info from
def createDataFrame(npz, nevents):
    df = digihits_info_to_df(npz, nevents)

    return df

# Add label to the DigiHits, this is supervised learning!
def addLabel(df):
    label = []
    for i in df['digi_hit_truehit_parent_trackID']:
        if -1 in i:
            label.append(0)
        else:
            label.append(1)

    df['label'] = label

    return df

# Edges Tensors
def edge_index(dat_id,
               event,
               numm_neigh,
               directed        = False,
               classic         = False,
               fully_connected = False,
               coord_names     = ['digi_hit_x', 'digi_hit_y', 'digi_hit_z'],
               torch_dtype     = torch.float):

    '''
    The function uses KDTree algorithm to create edge tensors for the graphs.
    Edges can be created based on N nearest neighbours, using the classic
    approach or connecting all of them using fully_connected arg.
    Number of nearest neighbours can be selected via the numm_neigh arg.
    To connect all, num_neigh = len(event) - 1 (to search for all the neighbors).

    Creates the edge index tensor, with shape [2, E] where E is the number of edges.
    It contains the index of the nodes that are connected by an edge.
    Also creates the edge features tensor, with shape [E, D] being D the number of features. In this case we add the distance.
    Also creates the edge weights tensor, with shape E: one weight assigned to each edge. In this case we use the inverse of the distance.
    '''

    def inve(dis): return 1/dis

    # Consider fully connected graph or N nearest neighbours
    if classic:
        num_neigh = numm_neigh
    if fully_connected:
        num_neigh = len(event) - 1

    hits = [tuple(x) for x in event[coord_names].to_numpy()]
    edges, edge_features, edge_weights = [], [], []

    # Build KD-Tree (Martin's approach, maybe this can be done differently)
    tree = KDTree(hits)

    # List to append the nodes we already looked into (to create directed graphs)
    passed_nodes = []
    try:
        for i, hit in enumerate(hits):
            # For each hit, get the N+1 nearest neighbors (first one is the hit itself)
            distances, indices = tree.query(hit, k=num_neigh+1)

            # For each neighbour, add edges
            for j, dis in zip(indices[1:], distances[1:]): # No self-connections
                if dis == 0: raise ValueError("Repeated hit {} in event {}".format(hit, dat_id))

                # Skip already passed nodes to create directed graphs
                if directed and np.isin(passed_nodes, j).any(): continue

                # Fill the edge lists
                edges.append([i, j])
                edge_features.append([dis])
                edge_weights.append(inve(dis))

            passed_nodes.append(i)

    except ValueError as e:
        print(f"Skipping event {dat_id} due to error: {e}")
        return None, None, None # Return None to signal this event should be skipped

    # Transform into the required tensors
    edges, edge_features, edge_weights = torch.tensor(edges, dtype=torch_dtype).T, torch.tensor(edge_features, dtype=torch_dtype), torch.tensor(edge_weights, dtype=torch_dtype)

    return edges, edge_features, edge_weights

def graph_Data(event,
               dat_id,
               num_neigh,
               features_n    = ['digi_hit_charge', 'digi_hit_time'],
               label_n       = 'label',
               directed      = False,
               classic       = False,
               all_connected = True,
               coord_names   = ['digi_hit_x', 'digi_hit_y', 'digi_hit_z'],
               torch_dtype   = torch.float):

    '''
    Creates for an event the Data PyTorch geometric object with the edges, edge features (distances), edge weights (inverse of distance),
    node features (digihit time and charge), label, number of nodes, and dataset ID.
    '''

    # Normalization function so input node features values aren't extreme
    def normalize(lst): return [i/np.max(lst) for i in lst]

    edges, edge_features, edge_weights = edge_index(dat_id, event, num_neigh, directed, classic, all_connected, coord_names, torch_dtype)

    # Node features normalized
    charge      = event['digi_hit_charge'].values.astype(np.float32)
    charge_norm = normalize(charge)
    time        = event['digi_hit_time'].values.astype(np.float32)
    time_norm   = normalize(time)

    # Node features information that pass to PyTorch Geometric
    nodes = torch.tensor(np.array([charge_norm, time_norm]).transpose())

    # Nodes labels information that pass to PyTorch Geometric
    label = torch.tensor(event[label_n].values)

    # Create Graph
    graph_data = Data(x=nodes, edge_index=edges, edge_attr=edge_features, edge_weight=edge_weights, y=label, num_nodes=len(nodes), dataset_id=dat_id, num_features=len(features_n))

    return graph_data

class GraphDataset(Dataset):
    def __init__(self, graph_data_list, num_features, num_classes):
        super(GraphDataset, self).__init__()
        self.graph_data_list = graph_data_list
        self._num_features   = num_features
        self._num_classes    = num_classes

    def len(self):
        return len(self.graph_data_list)

    def get(self, idx):
        return self.graph_data_list[idx]

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_classes(self):
        return self._num_classes

def graphDataset(file,
                 df,
                 features_n    = ['digi_hit_charge', 'digi_hit_time'],
                 label_n       = 'label',
                 num_neigh     = 5,
                 num_classes   = 2,
                 directed      = False,
                 classic       = False,
                 all_connected = True,
                 coord_names   = ['digi_hit_x', 'digi_hit_y', 'digi_hit_z'],
                 torch_dtype   = torch.float):
    '''
    For a file, it creates a dataset with all the events in their input in the GNN form
    '''

    # Get the .npz file name
    filename      = os.path.splitext(os.path.basename(file))[0]
    # Initialize the DataSet list
    dataset       = []
    # Get only those events with DigiHits
    nonZeroEvents = df.groupby('event_id').size().index.to_numpy()

    # Create the Graph for every event and append it to the dataset list
    for ev in nonZeroEvents:
        event      = df[df['event_id'].values == ev]
        dat_id     = ev
        graph_data = graph_Data(event,
                               dat_id,
                               num_neigh,
                               features_n    = features_n,
                               label_n       = label_n,
                               directed      = directed,
                               classic       = classic,
                               all_connected = all_connected,
                               coord_names   = coord_names,
                               torch_dtype   = torch_dtype)

        # In order to avoid fraphs where edges don't exist
        # check if graph_data is not None before proceeding
        if graph_data is not None and graph_data.edge_index is not None and graph_data.edge_index.numel() > 0:
            graph_data.fnum = filename+f'_event{ev}'
            dataset.append(graph_data)

    return GraphDataset(graph_data_list=dataset, num_features=len(features_n), num_classes=num_classes)
