import tensorflow as tf

from argparse import ArgumentParser

from src.data.dataset import FloorPlanDataset, FloorPlanDataType
from src.models.corner_detector import CornerDetector
from src.trainers.corner_trainer import CornerDetectorTrainer
from src.utils.config import ConfigParser
from src.utils.utils import set_gpu_growth


def train_corners(config_file, params):

    config = ConfigParser(config_file, params).config

    floor_plan_dataset = FloorPlanDataset(config.data, data_type=FloorPlanDataType.TFRECORD)
    train_dataset = floor_plan_dataset.generate_dataset('train', max_samples=-1)

    model = CornerDetector(config.model.input_dim, config.model.output_dim, None,
                           config.data.width, config.data.height)
    corner_trainer = CornerDetectorTrainer(config.train, model, train_dataset)
    print(model.summary())
    corner_trainer.train()


if __name__ == '__main__':
    set_gpu_growth()

    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to the config file. Default file: config/corner_train.yaml',
                        default='config/corner_train.yaml')
    args = parser.parse_args()
    train_corners(args.config, None)
