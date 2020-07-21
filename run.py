import argparse
import warnings

from src.scripts.glo_train import train_p2p_glo
from src.scripts.corner_train import train_corners
from src.scripts.transform import transform_vectors
from src.scripts.summary import model_summary


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    help='Path to the config file. Default is config/corner_train.yml.',
                    default='config/corner_train.yaml', required=False)

parser.add_argument('-sw', '--show_warnings',
                    help='Toggle this command if you want to show warnings',
                    action='store_true')

sub_parsers = parser.add_subparsers(help='', dest='mode')

transformer_parser = sub_parsers.add_parser(
    'transform', help='Use this to transform vectors to trainable data')
corner_train_parser = sub_parsers.add_parser(
    'corner_train', help='Use this to start training a model to detect corners')
p2p_glo_train_parser = sub_parsers.add_parser(
    'p2p_glo_train', help='Use this to start training a pix2pix model with glo ')
summary_parser = sub_parsers.add_parser(
    'summary', help='Use this to show the summary of the models used')

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
    elif args.mode == 'transform':
        print('Transforming vectors to data')
        transform_vectors(args.config, args.__dict__)
    elif args.mode == 'summary':
        print('Printing the summary of the models')
        model_summary()
    else:
        print('Bye!')

