from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.random_shape_test import RandomShapeTester
from utils.config_parser import Config
from utils.utils import set_gpu_growth

set_gpu_growth()

config = Config('./config/evaluate.yaml')
DATA_DIR = config.get_string('data', './datasets/tfrecords')
WIDTH = config.get_int('width', 128)
HEIGHT = config.get_int('height', 128)
LAT_DIM = config.get_int('latent_dimensions', 32)
GEN_CKPT = config.get_string('gen_ckpt')
NUM_SAMPLES = config.get_int('num_samples', 3)

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('test', max_samples=1)

evaluator = RandomShapeTester(dataset, LAT_DIM, GEN_CKPT)
evaluator.evaluate(NUM_SAMPLES)
