from argparse import ArgumentParser

import tensorflow as tf

from src.data.dataset import FloorPlanDataset, FloorPlanDataType
from src.models.evaluator import FloorPlanGenerator
from src.utils.config import ConfigParser
from src.utils.config_parser import Config
from src.utils.utils import set_gpu_growth

set_gpu_growth()


def evaluate(config, params):
    config = ConfigParser(config, params).config
    # config = Config(config)

    # lat_dim = config.get_int('latent_dimensions', 32)
    # meta_dim = config.get_int('meta_dimensions', 14)
    # gen_ckpt = config.get_string('gen_ckpt')
    # corner_ckpt = config.get_string('corner_ckpt')
    # num_samples = config.get_int('num_samples', 3)

    floor_plan_dataset = FloorPlanDataset(config.data, data_type=FloorPlanDataType.TFRECORD)

    dataset = floor_plan_dataset.generate_dataset('test', max_samples=config.test.num_samples)

    evaluator = FloorPlanGenerator(dataset, config.model.lat_dim, config.model.meta_dim, config.test.gen_ckpt, config.test.corner_ckpt)

    dataset = dataset.batch(1).take(1)

    for index, (cooling, corner_mask, door_count, door_mask, entrance_mask, heating, room_mask, room_types, shape_mask, wall_count, wall_mask, window_count, window_mask) in dataset.enumerate():

       # dc = tf.expand_dims(door_count, 1)
       # wic = tf.expand_dims(window_count, 1)
       # cl = tf.expand_dims(cooling, 1)
       # ht = tf.expand_dims(heating, 1)
       #
       # meta_input = tf.concat([room_types, dc, wic, cl, ht], axis=1)
       #
        meta_input = tf.concat([room_types, door_count, window_count, cooling, heating], axis=0)

        #walls, doors, windows, rooms, corners, shape,
        evaluator.reconstruct(wall_mask, door_mask, window_mask, room_mask, corner_mask, shape_mask, room_types, meta_input, str(index),
                              save=True, show_walls=True, show_doors=True,
                              show_windows=True, show_rooms=True, show_corners=True,
                              show_shape=True, show_reconstructed=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to the config file. Default file: config/evaluate.yaml',
                        default='./config/evaluate.yaml')
    args = parser.parse_args()
    evaluate(args.config, args.__dict__)
