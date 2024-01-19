from torch.cuda import set_device

from helpers.parse_arguments import parse_arguments
from experiments import Experiment

if __name__ == '__main__':
    args = parse_arguments()
    if args.gpu is not None:
        set_device(args.gpu)
    Experiment(args=args).run()
