import argparse
import warnings

from src.scripts.glo_train import train_p2p_glo
from src.scripts.corner_train import train_corners

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    help='Path to the config file. Default is config/bird_train.yml.',
                    default='config/corner_train.yaml', required=False)

parser.add_argument('-sw', '--show_warnings',
                    help='Toggle this command if you want to show warnings',
                    action='store_true')

sub_parsers = parser.add_subparsers(help='', dest='mode')

corner_train_parser = sub_parsers.add_parser(
    'corner_train', help='Use this to start training a model to detect corners')
p2p_glo_train_parser = sub_parsers.add_parser(
    'p2p_glo_train', help='Use this to start training a pix2pix model with glo ')

args = parser.parse_args()

if not args.show_warnings:
    warnings.filterwarnings('ignore')

if __name__ == '__main__':

    if args.mode == 'corner_train':
        print('Starting the training of a corner detector model........')
        train_corners(args.config, args.__dict__)
    elif args.mode == 'p2p_glo_train':
        print('Starting the training of a p2p model using glo........')
        train_p2p_glo(args.config, args.__dict__)
    else:
        print('Bye!')

