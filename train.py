from argparse import ArgumentParser
import tensorflow as tf

from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.trainer import GANTrainer

tf.enable_eager_execution()

parser = ArgumentParser()
parser.add_argument("-i", "--data", help='Path to the directory where the data is stored',
                    default='/home/harikatam/TUM/sose2019/IDP/datasets/tfrecords')
parser.add_argument("-s", "--summary", help='Path to the directory where the summaries should be stored',
                    default='./summaries')
parser.add_argument("-c", "--ckpt", help='Path to the directory where the checkpoints should be stored',
                    default='./checkpoints')
parser.add_argument("-e", "--epochs", help='Path to the directory where the checkpoints should be stored',
                    type=int, default=20)
parser.add_argument("-b", "--batch", help='Path to the directory where the checkpoints should be stored',
                    type=int, default=16)

args = vars(parser.parse_args())

DATA_DIR = args['data']
WIDTH = 256
HEIGHT = 256

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_train_dataset('train', max_samples=-1,
                                                    include_walls=True, include_doors=True,
                                                    include_windows=True, include_rooms=True,
                                                    include_shape=True, include_corners=False)

gan_trainer = GANTrainer(dataset, WIDTH, HEIGHT, save_summary=True, summary_dir=args['summary'],
                         save_gen_ckpt=True, save_disc_ckpt=True, ckpt_dir=args['ckpt'])

EPOCHS = args['epochs']
BATCH_SIZE = args['batch']
NUM_SAMPLES = 4000

gen_config = {
    'optimizer': 'adam',
    'lr': 2e-4
}

disc_config = {
    'optimizer': 'adam',
    'lr': 2e-4
}

gan_trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
                  shuffle=False, coeff=100, load_gen_ckpt=None, load_disc_path=None,
                  gen_config=gen_config, disc_config=disc_config)
