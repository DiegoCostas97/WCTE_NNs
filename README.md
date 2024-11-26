# WCTE Analysis Using Neural Networks

This is a first attempt of coding, training and making predictions with a Graph Neural Network for separating Dark Noise (background) hits from Signal hits in WCTE. 

The idea behind is that every event is represented as a graph, in which the nodes are the hits in the PMTs. Nodes have two main features, charge and time, and the edges are the physical distance between PMTs in the detector. This can be upgraded including more features to the nodes, or even adding weights to the edges.

# Current Status
We now can train the net using the `main.py` script and the training info and metrics are written to the Tensorboard logger. You need to change the parameters in the config file dedicated to the model you are using.

# How to
1. At the moment, first you need to create a dataset for your data. You can do that running `python3 scripts/create_dataset.py -conf scripts/config_files/dataset_creation.conf` 
2. Then you need to check `scripts/config_files/GAT.conf` and tune the parameters you want to use.
3. Then run `main.py` doing: `python3 main.py -conf scripts/config_files/GAT.conf -a train`.
4. Once the training is completed and tensorboard file created, you can open the web application doing: `tensorboard --logdir=.` from the directory where the tensorboard file is located.

# To-Do
- Add prediction to the `main.py` function.
- ~Add a dedicated script that creates the graph dataset and stores it somewhere the net can read it from.~
- In general, improve performance of the net.
