from data.dataset import FloorPlanDataset, FloorPlanDataType
from models.trainer import GANTrainer

DATA_DIR = '/home/harikatam/TUM/sose2019/IDP/datasets/'
WIDTH = 256
HEIGHT = 256

floor_plan_dataset = FloorPlanDataset(data_dir=DATA_DIR, width=WIDTH, height=HEIGHT,
                                      data_type=FloorPlanDataType.HDF5)

dataset = floor_plan_dataset.generate_train_dataset('h5py', include_walls=True, include_doors=True,
                                                    include_windows=True, include_rooms=True,
                                                    include_shape=True, include_corners=False)

gan_trainer = GANTrainer(dataset, WIDTH, HEIGHT, save_summary=True, summary_dir='./summaries',
                         save_gen_ckpt=True, save_disc_ckpt=True, ckpt_dir='./checkpoints')

EPOCHS = 10
BATCH_SIZE = 32
NUM_SAMPLES = 50

gen_config = {
    'optimizer': 'adam',
    'lr': 2e-4
}

disc_config = {
    'optimizer': 'adam',
    'lr': 2e-4
}

gan_trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,
                  shuffle=True, coeff=100, load_gen_ckpt=None, load_disc_path=None,
                  gen_config=gen_config, disc_config=disc_config)
