from argparse import ArgumentParser

from data.data_transformer import VectorToImageTransformer, TransformerConfig
from utils.utils import set_gpu_growth

set_gpu_growth()

parser = ArgumentParser()
parser.add_argument("-i", "--input", help='Path to the directory where the data is stored',
                    default='../public_datasets/vectors/')
parser.add_argument("-o", "--output", help='Path to the directory where the data is stored',
                    default='./datasets/tfrecords/train')
parser.add_argument("-t", "--type", help='Type of the data (tfrecord or hdf5)',
                    default='tfrecord')
args = vars(parser.parse_args())

config = TransformerConfig(wall_thickness=3,
                           window_thickness=3,
                           door_thickness=3,
                           inp_dir=args['input'],
                           out_width=256,
                           out_height=256,
                           out_format=args['type'],
                           out_dir=args['output'])

vector_to_image_transformer = VectorToImageTransformer(config, num_images=6000)
vector_to_image_transformer.transform_vectors_to_images()
