from argparse import ArgumentParser

from helpers.dataset_classes.dataset import DataSet
from helpers.model import ModelType
from helpers.classes import Pool
from helpers.encoders import PosEncoder


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", default=DataSet.roman_empire, type=DataSet.from_string,
                        choices=list(DataSet), required=False)
    parser.add_argument("--pool", dest="pool", default=Pool.NONE, type=Pool.from_string,
                        choices=list(Pool), required=False)

    # gumbel
    parser.add_argument("--learn_temp", dest="learn_temp", default=False, action='store_true', required=False)
    parser.add_argument("--temp_model_type", dest="temp_model_type", default=ModelType.LIN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument("--tau0", dest="tau0", default=0.5, type=float, required=False)
    parser.add_argument("--temp", dest="temp", default=0.01, type=float, required=False)

    # optimization
    parser.add_argument("--max_epochs", dest="max_epochs", default=3000, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=32, type=int, required=False)
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float, required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.2, type=float, required=False)

    # env cls parameters
    parser.add_argument("--env_model_type", dest="env_model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument("--env_num_layers", dest="env_num_layers", default=3, type=int, required=False)
    parser.add_argument("--env_dim", dest="env_dim", default=128, type=int, required=False)
    parser.add_argument("--skip", dest="skip", default=False, action='store_true', required=False)
    parser.add_argument("--batch_norm", dest="batch_norm", default=False, action='store_true', required=False)
    parser.add_argument("--layer_norm", dest="layer_norm", default=False, action='store_true', required=False)
    parser.add_argument("--dec_num_layers", dest="dec_num_layers", default=1, type=int, required=False)
    parser.add_argument("--pos_enc", dest="pos_enc", default=PosEncoder.NONE,
                        type=PosEncoder.from_string, choices=list(PosEncoder), required=False)

    # policy cls parameters
    parser.add_argument("--act_model_type", dest="act_model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument("--act_num_layers", dest="act_num_layers", default=1, type=int, required=False)
    parser.add_argument("--act_dim", dest="act_dim", default=16, type=int, required=False)

    # reproduce
    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)

    # dataset dependant parameters 
    parser.add_argument("--fold", dest="fold", default=None, type=int, required=False)

    # optimizer and scheduler
    parser.add_argument("--weight_decay", dest="weight_decay", default=0, type=float, required=False)
    ## for steplr scheduler only
    parser.add_argument("--step_size", dest="step_size", default=None, type=int, required=False)
    parser.add_argument("--gamma", dest="gamma", default=None, type=float, required=False)
    ## for cosine with warmup scheduler only
    parser.add_argument("--num_warmup_epochs", dest="num_warmup_epochs", default=None, type=int, required=False)

    return parser.parse_args()
