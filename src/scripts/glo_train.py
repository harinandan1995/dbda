import json

from argparse import ArgumentParser

from src.data.dataset import FloorPlanDataset, FloorPlanDataType
from src.models.generator import Generator
from src.trainers.glo import GLO
from src.utils.config import ConfigParser
from src.utils.utils import set_gpu_growth

set_gpu_growth()


def train_p2p_glo(config_path, params):

    config = ConfigParser(config_path, params).config
    print(json.dumps(config, indent=3))

    floor_plan_dataset = FloorPlanDataset(config.data, data_type=FloorPlanDataType.TFRECORD)
    train_dataset = floor_plan_dataset.generate_dataset('train', max_samples=-1)

    model = Generator(config.model.input_dim, config.model.output_dim,
                      config.model.lat_dim, config.model.meta_dim, None,
                      config.data.width, config.data.height)
    print(model.summary())
    gan_trainer = GLO(config.train, model, train_dataset)

    gan_trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to the config file. Default file: config/glo_train.yaml',
                        default='config/glo_train.yaml')
    args = parser.parse_args()

    train_p2p_glo(args.config, None)
