import os
import torch
import sys

sys.path.append("/home/usc/ie/dcr/hk/ml/hit_ana_gnn/utils")
sys.path.append("/home/usc/ie/dcr/hk/ml/hit_ana_gnn/data_loader")
sys.path.append("/home/usc/ie/dcr/hk/ml/hit_ana_gnn/networks")

from argparse import ArgumentParser, Namespace

from train_utils      import train_net, predict_gen
from architectures import GAT

def is_file(parser, arg):
    """
    Check if the file passed as argument exists.
    """
    if not os.path.exists(arg):
        parser.error("The file %s does not exist" % arg)
    else:
        return arg

def is_valid_action(parser, arg):
    """
    Check if the action passed as argument is valid.
    """
    if not arg in ['train', 'predict']:
        parser.error("The action %s is not allowed!" % arg)
    else:
        return arg

def get_params(confname):
    """
    Get Parameters from Confid file.
    """
    file_name  = os.path.expandvars(confname)
    parameters = {}

    with open(file_name, 'r') as conf_file:
        exec(conf_file.read(), parameters)
    return Namespace(**parameters)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    torch.backends.cudnn.enables = True
    torch.backends.cudnn.benchmark = True

    parser = ArgumentParser(description="Parameters for models")
    parser.add_argument("-conf", dest="confname", required=True, help="Input file with parameters",
                        metavar="FILE", type=lambda x:is_file(parser, x))
    parser.add_argument("-a", dest="action", required=True, help="Action for the model to perform: train or predict",
                        type=lambda x:is_valid_action(parser, x))

    args = parser.parse_args()
    confname = args.confname
    action   = args.action
    params   = get_params(confname)

    if params.netarch == 'GAT':
        model = GAT(params.num_features,
                    params.hidden_channels,
                    params.dropout_gat,
                    params.dropout_fc)

    print("Net constructed")

    dataset = torch.load(params.data_file)
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_data, valid_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    if action == "train":
        criterion_class = getattr(torch.nn, params.LossType)
        criterion = criterion_class()

        optimizer_class = getattr(torch.optim, params.OptimType)
        optimizer = optimizer_class(model.parameters(),
                                    lr = params.lr)

        if params.SchedulerType == "None":
            scheduler = None
        else:
            scheduler_class = getattr(torch.optim.lr_scheduler, params.SchedulerType)
            scheduler = scheduler_class(optimizer,
                                        factor   = params.reduce_lr_factor,
                                        patience = params.reduce_lr_patience,
                                        min_lr   = params.reduce_lr_min_lr)

        train_net(nepoch           = params.nepoch,
                  train_dataset    = train_data,
                  valid_dataset    = valid_data,
                  train_batch_size = params.train_batch,
                  valid_batch_size = params.valid_batch,
                  model            = model,
                  optimizer        = optimizer,
                  criterion        = criterion,
                  scheduler        = scheduler,
                  checkpoint_dir   = params.checkpoint_dir,
                  tensorboard_dir  = params.tensorboard_dir)
