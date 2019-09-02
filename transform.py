from argparse import ArgumentParser
from data.data_transformer import VectorToImageTransformer, TransformerConfig

parser = ArgumentParser()
parser.add_argument("-i", "--input", help='Path to the directory where the data is stored',
                    default='./datasets/vectors')
parser.add_argument("-o", "--output", help='Path to the directory where the data is stored',
                    default='./datasets/tfrecords')
parser.add_argument("-t", "--type", help='Type of the data (tfrecord or hdf5)',
                    default='tfrecord')
args = vars(parser.parse_args())

config = TransformerConfig(wall_thickness=3,
                           window_thickness=2,
                           door_thickness=2,
                           inp_dir=args['input'],
                           out_width=256,
                           out_height=256,
                           out_format=args['type'],
                           out_dir=args['output'])

vector_to_image_transformer = VectorToImageTransformer(config, num_images=5000)
vector_to_image_transformer.transform_vectors_to_images()
