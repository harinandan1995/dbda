from argparse import ArgumentParser

from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.evaluator import FloorPlanGenerator
from utils.utils import set_gpu_growth

set_gpu_growth()

parser = ArgumentParser()
parser.add_argument("-i", "--data", help='Path to the directory where the data is stored',
                    default='./datasets/tfrecords')
parser.add_argument("-c", "--ckpt", help='Path to the directory where the checkpoints should be stored',
                    default='./checkpoints/20191025/211207/gen_20191025_233943_22.h5')

args = vars(parser.parse_args())

DATA_DIR = args['data']

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('val', max_samples=-1)

evaluator = FloorPlanGenerator(dataset, args['ckpt'])
evaluator.evaluate(5)
