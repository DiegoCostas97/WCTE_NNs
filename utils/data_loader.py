# Some imports, not really important what they do

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np
import networkx          as nx

import torch
import sys

# Import Diego's tools
sys.path.append("/home/usc/ie/dcr/hk/ambe_analysis/paquetes")
from npz_to_df import digihits_info_to_df

from scipy.spatial        import KDTree
from torch_geometric.data import Data, Dataset

# Read .npz data
# To produce this .npz from wcsim_output.root, use event_dump.py from DataTools
npz = '/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/nicfVec_5kHzDR00-1350_7Th200ns-400+950.npz'

# Number of events simulated
nevents = 30000

# Creation of the DataFrame the DataLoader is reading the info from
df_digiHits = df_digihits_info_to_df(npz, nevents)

# Add label to the DigiHits, this is supervised learning!
label = []
for i in df_digiHits['digi_hit_truehit_parent_trackID']:
    if -1 in i:
        label.append(0)
    else:
        label.append(1)

df_digiHits['label'] = label

# Edges Tensors
def edge_index(dat_id,
               event,
               numm_neigh,
               directed=False,
               classic=False,
               fully_connected=False,
               coords_names = ['digi_hit_x', 'digi_hit_y', 'digi_hit_z'],
               torch_dtype = torch.float):

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



