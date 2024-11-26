import torch
import sys

sys.path.append("/home/usc/ie/dcr/hk/ml/hit_ana_gnn")

from utils.data_loader import createDataFrame, addLabel, graphDataset

from scripts.main import is_file, get_params

from argparse import ArgumentParser, Namespace

if __name__ == "__main__":
    parser.add_argument("-conf", dest="confname", required=True, help="Input file with parameters",
                        metavar="FILE", type=lambda x:is_file(parser, x))

    args     = parser.parse_args()
    confname = args.confname
    params   = get_params(confname)

    df = createDataFrame(params.npz, params.nevents)
    df = addLabel(df)

    gnnDataset = graphDataset(file          = params.npz,
                              df            = params.df,
                              num_neigh     = params.num_neigh,
                              num_classes   = params.num_classes,
                              directed      = params.directed,
                              classic       = params.classic,
                              all_connected = params.all_connected)

    torch.save(gnnDataset, params.output_file)
