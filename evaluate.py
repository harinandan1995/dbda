from argparse import ArgumentParser

from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.evaluator import FloorPlanGenerator

parser = ArgumentParser()
parser.add_argument("-i", "--data", help='Path to the directory where the data is stored',
                    default='./datasets/tfrecords')
parser.add_argument("-c", "--ckpt", help='Path to the directory where the checkpoints should be stored',
                    default='./checkpoints/20190906/174818/gen_20190906_212621_28.h5')

args = vars(parser.parse_args())

DATA_DIR = args['data']
WIDTH = 256
HEIGHT = 256

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('val', max_samples=-1,
                                              include_walls=True, include_doors=True,
                                              include_windows=True, include_rooms=True,
                                              include_shape=True, include_corners=False)

evaluator = FloorPlanGenerator(dataset, args['ckpt'], WIDTH, HEIGHT)
evaluator.evaluate(5)
