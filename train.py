import os
from argparse import ArgumentParser

from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.trainer import GANTrainer
from utils.utils import set_gpu_growth

set_gpu_growth()

parser = ArgumentParser()
parser.add_argument("-i", "--data", help='Path to the directory where the data is stored',
                    default='./datasets/tfrecords')
parser.add_argument("-s", "--summary", help='Path to the directory where the summaries should be stored',
                    default='./summaries')
parser.add_argument("-c", "--ckpt", help='Path to the directory where the checkpoints should be stored',
                    default='./checkpoints')
parser.add_argument("-e", "--epochs", help='Path to the directory where the checkpoints should be stored',
                    type=int, default=20)
parser.add_argument("-b", "--batch", help='Path to the directory where the checkpoints should be stored',
                    type=int, default=8)

args = vars(parser.parse_args())

DATA_DIR = args['data']
WIDTH = 256
HEIGHT = 256

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('train', max_samples=-1)

CKPT_DIR = './checkpoints/20191025/211207'
GEN_CKPT = os.path.join(CKPT_DIR, 'gen_20191025_233943_22.h5')
DISC_CKPT = os.path.join(CKPT_DIR, 'disc_20191025_233943_22.h5')

gan_trainer = GANTrainer(dataset, WIDTH, HEIGHT, save_summary=True, summary_dir=args['summary'],
                         save_gen_ckpt=True, save_disc_ckpt=True, ckpt_dir=args['ckpt'])

EPOCHS = args['epochs']
BATCH_SIZE = args['batch']
NUM_SAMPLES = 5200

gen_config = {
    'optimizer': 'adam',
    'lr': 8e-4
}

disc_config = {
    'optimizer': 'adam',
    'lr': 4e-4
}

gan_trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
                  shuffle=False, coeff=5, load_gen_ckpt=None, load_disc_path=None,
                  gen_config=gen_config, disc_config=disc_config)
