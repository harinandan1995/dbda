from argparse import ArgumentParser

from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.evaluator import FloorPlanGenerator
from utils.utils import set_gpu_growth
from utils.config_parser import Config

set_gpu_growth()

config = Config('./config/evaluate.yaml')
DATA_DIR = config.get_string('data', './datasets/tfrecords')
WIDTH = config.get_int('width', 128)
HEIGHT = config.get_int('height', 128)
GEN_CKPT = config.get_string('gen_ckpt')
NUM_SAMPLES = config.get_int('num_samples', 3)

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('val', max_samples=-1)

evaluator = FloorPlanGenerator(dataset, GEN_CKPT)
evaluator.evaluate(NUM_SAMPLES)
