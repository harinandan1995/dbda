from argparse import ArgumentParser

from data.dataset import FloorPlanDataset, FloorPlanDataType
from trainers.glo_trainer import GLOTrainer
from utils.config_parser import Config
from utils.utils import set_gpu_growth

set_gpu_growth()

parser = ArgumentParser()
parser.add_argument('--config', help='Path to the config file. Default file: config/glo_train.yaml',
                    default='./config/glo_train.yaml')
args = parser.parse_args()

config = Config(args.config)


config = Config('./config/glo_train.yaml')

# Data configs
DATA_DIR = config.get_string('data', './datasets/tfrecords')
WIDTH = config.get_int('width', 256)
HEIGHT = config.get_int('height', 256)
LAT_DIM = config.get_int('latent_dimensions', 8)
GEN_CKPT = config.get_string('gen_ckpt', '')
CKPT_DIR = config.get_string('ckpt_dir', './checkpoints')
SUMMARY_DIR = config.get_string('summary_dir', './summaries')

# Hyper parameters
EPOCHS = config.get_int('epochs', 2)
BATCH_SIZE = config.get_int('batch_size', 8)
NUM_SAMPLES = config.get_int('num_samples', 1000)
GEN_LR = config.get_float('gen_lr')
LAT_LR = config.get_float('lat_lr')
LAT_ITER = config.get_int('latent_iterations', 20)
GEN_ITER = config.get_int('generator_iterations', 5)

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('train', max_samples=-1)

gan_trainer = GLOTrainer(dataset, WIDTH, HEIGHT, LAT_DIM, save_summary=True, summary_dir=SUMMARY_DIR,
                         save_gen_ckpt=True, ckpt_dir=CKPT_DIR)

gen_config = {
    'optimizer': 'adam',
    'lr': GEN_LR
}

lat_config = {
    'optimizer': 'adam',
    'lr': LAT_LR
}

gan_trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
                  lat_iter=LAT_ITER, gen_iter=GEN_ITER, shuffle=True,
                  load_gen_ckpt=GEN_CKPT, gen_config=gen_config, lat_config=lat_config)
