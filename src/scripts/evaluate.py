from argparse import ArgumentParser

import tensorflow as tf
import json

from src.data.dataset import FloorPlanDataset, FloorPlanDataType
from src.models.evaluator import FloorPlanGenerator
from src.utils.config import ConfigParser
from src.utils.config_parser import Config
from src.utils.utils import set_gpu_growth

set_gpu_growth()


def evaluate(config, params):
    
    config = ConfigParser(config, params).config
    print(json.dumps(config, indent=3))
    
    floor_plan_dataset = FloorPlanDataset(config.data, data_type=FloorPlanDataType.TFRECORD)
    dataset = floor_plan_dataset.generate_dataset('test', max_samples=config.test.num_samples)

    evaluator = FloorPlanGenerator(dataset, config.model.lat_dim, config.model.meta_dim, config.test.gen_ckpt, config.test.corner_ckpt, config.test.iterations)

    dataset = dataset.batch(1)

    for step, data in dataset.enumerate():
        
        dc = tf.expand_dims(data["door_count"], 1)
        wic = tf.expand_dims(data["window_count"], 1)
        cl = tf.expand_dims(data["cooling"], 1)
        ht = tf.expand_dims(data["heating"], 1)


        meta_input = tf.concat([data["room_types"], dc, wic, cl, ht], axis=1)

        if config.test.type == 'one_to_many':
            evaluator.get_possible_plans(data["shape_mask"], meta_input, number_of_outputs=config.test.num_outputs)
        else:
            evaluator.reconstruct(data["wall_mask"], data["door_mask"], data["window_mask"], data["room_mask"], 
                                data["corner_mask"], data["shape_mask"], data["room_types"],
                                meta_input, str(step), save=True, show_walls=True, show_doors=True,
                                show_windows=True, show_rooms=True, show_corners=True,
                                show_shape=True, show_reconstructed=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to the config file. Default file: config/evaluate.yaml',
                        default='./config/evaluate.yaml')
    args = parser.parse_args()
    evaluate(args.config, args.__dict__)
