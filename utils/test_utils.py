from data_loader import createDataFrame
from data_loader import addLabel
from data_loader import graphDataset

from plot_utils import visualize_graph

import matplotlib.pyplot as plt

import time


# Read .npz data
# To produce this .npz from wcsim_output.root, use event_dump.py from DataTools
npz = '/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/nicfVec_5kHzDR00-1350_7Th200ns-400+950.npz'

# Number of events simulated
nevents = 30000

print("Creating DigiHits DataFrame...")
df = createDataFrame(npz, nevents)
print("DigiHits DataFrame Created!\n")
time.sleep(2)

print("Adding Labels To DigiHits...")
df = addLabel(df)
print("Labels Added!\n")
time.sleep(2)

print("Transforming Data Into Graphs and Creating Graphs Dataset...")
gnnDataset = graphDataset(file=npz, df=df, num_neigh=5, num_classes=2, directed=False, classic=True, all_connected=False)
print("\nGraph Dataset Created!")

# Plot one graph
graph = gnnDataset[10]
visualize_graph(graph, color=graph.y)
plt.show()
