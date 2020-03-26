from argparse import ArgumentParser

from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.evaluator import FloorPlanGenerator
from utils.config_parser import Config
from utils.utils import set_gpu_growth

set_gpu_growth()

parser = ArgumentParser()
parser.add_argument('--config', help='Path to the config file. Default file: config/evaluate.yaml',
                    default='./config/evaluate.yaml')
args = parser.parse_args()

config = Config(args.config)

DATA_DIR = config.get_string('data', './datasets/tfrecords')
WIDTH = config.get_int('width', 256)
HEIGHT = config.get_int('height', 256)
LAT_DIM = config.get_int('latent_dimensions', 32)
GEN_CKPT = config.get_string('gen_ckpt')
CORNER_CKPT = config.get_string('corner_ckpt')
NUM_SAMPLES = config.get_int('num_samples', 3)

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('test', max_samples=NUM_SAMPLES)

evaluator = FloorPlanGenerator(dataset, LAT_DIM, GEN_CKPT, CORNER_CKPT)
evaluator.evaluate(NUM_SAMPLES)
