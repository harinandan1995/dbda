from argparse import ArgumentParser

from data.data_transformer import VectorToImageTransformer, TransformerConfig
from utils.utils import set_gpu_growth
from utils.config_parser import Config

set_gpu_growth()

config = Config('./config/transform.yaml')

IN_DIR = config.get_string('input_dir')
OUT_DIR = config.get_string('output_dir')
OUT_TYPE = config.get_string('output_type', 'tfrecord')

WIDTH = config.get_int('width', 256)
HEIGHT = config.get_int('height', 256)
WALL_THICKNESS = config.get_int('wall_thickness', 2)
WINDOW_THICKNESS = config.get_int('window_thickness', 2)
DOOR_THICKNESS = config.get_int('door_thickness', 2)
CORNER_THICKNESS = config.get_int('corner_thickness', 1)
NUM_OUTPUT_IMAGES = config.get_int('num_output_images', 1000)
SHUFFLE = config.get_bool('shuffle', True)

config = TransformerConfig(wall_thickness=WALL_THICKNESS,
                           window_thickness=WINDOW_THICKNESS,
                           door_thickness=DOOR_THICKNESS,
                           corner_thickness=CORNER_THICKNESS,
                           inp_dir=IN_DIR,
                           out_width=WIDTH,
                           out_height=HEIGHT,
                           out_format=OUT_TYPE,
                           out_dir=OUT_DIR)

vector_to_image_transformer = VectorToImageTransformer(
                                    config, num_images=NUM_OUTPUT_IMAGES,
                                    shuffle=SHUFFLE)
vector_to_image_transformer.transform_vectors_to_images()
