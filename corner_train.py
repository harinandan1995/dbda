from argparse import ArgumentParser

from data.dataset import FloorPlanDataset, FloorPlanDataType
from trainers.corner_detector_trainer import CornerDetectorTrainer
from utils.config_parser import Config
from utils.utils import set_gpu_growth

set_gpu_growth()

parser = ArgumentParser()
parser.add_argument('--config', help='Path to the config file. Default file: config/corner_train.yaml',
                    default='./config/corner_train.yaml')
args = parser.parse_args()

config = Config(args.config)

# Data configs
DATA_DIR = config.get_string('data', './datasets/tfrecords')
WIDTH = config.get_int('width', 256)
HEIGHT = config.get_int('height', 256)
MODEL_CKPT = config.get_string('model_ckpt', '')
CKPT_DIR = config.get_string('ckpt_dir', './checkpoints')
SUMMARY_DIR = config.get_string('summary_dir', './summaries')

# Hyper parameters
EPOCHS = config.get_int('epochs', 2)
BATCH_SIZE = config.get_int('batch_size', 8)
NUM_SAMPLES = config.get_int('num_samples', 1000)
LR = config.get_float('lr')

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)
dataset = floor_plan_dataset.generate_dataset('train', max_samples=-1)

gan_trainer = CornerDetectorTrainer(dataset, WIDTH, HEIGHT, save_summary=True, summary_dir=SUMMARY_DIR,
                                    save_model_ckpt=True, ckpt_dir=CKPT_DIR)

model_config = {
    'optimizer': 'adam',
    'lr': LR
}

gan_trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
                  shuffle=False, model_ckpt=MODEL_CKPT, model_config=model_config)
