from data.dataset import FloorPlanDataset, FloorPlanDataType
from trainers.p2p_trainer import Pix2PixTrainer
from utils.utils import set_gpu_growth
from utils.config_parser import Config

set_gpu_growth()

config = Config('./config/p2p_train.yaml')

# Data configs
DATA_DIR = config.get_string('data', './datasets/tfrecords')
WIDTH = config.get_int('width', 128)
HEIGHT = config.get_int('height', 128)
LAT_DIM = config.get_int('latent_dimensions', 8)
GEN_CKPT = config.get_string('gen_ckpt', '')
DISC_CKPT = config.get_string('disc_ckpt', '')
CKPT_DIR = config.get_string('ckpt_dir', './checkpoints')
SUMMARY_DIR = config.get_string('summary_dir', './summaries')

# Hyper parameters
EPOCHS = config.get_int('epochs', 2)
BATCH_SIZE = config.get_int('batch_size', 8)
NUM_SAMPLES = config.get_int('num_samples', 1000)
GEN_LR = config.get_float('gen_lr')
DISC_LR = config.get_float('disc_lr')
COEFF = config.get_float('loss_coeff', 5)

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.TFRECORD)

dataset = floor_plan_dataset.generate_dataset('train', max_samples=-1)

gan_trainer = Pix2PixTrainer(dataset, WIDTH, HEIGHT, LAT_DIM, save_summary=True, summary_dir=SUMMARY_DIR,
                         save_gen_ckpt=True, save_disc_ckpt=True, ckpt_dir=CKPT_DIR)

gen_config = {
    'optimizer': 'adam',
    'lr': GEN_LR
}

disc_config = {
    'optimizer': 'adam',
    'lr': DISC_LR
}

gan_trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
                  shuffle=False, coeff=COEFF, load_gen_ckpt=GEN_CKPT, load_disc_path=DISC_CKPT,
                  gen_config=gen_config, disc_config=disc_config)
