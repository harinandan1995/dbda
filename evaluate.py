from argparse import ArgumentParser

import tensorflow as tf

from src.data.dataset import FloorPlanDataset, FloorPlanDataType
from src.models.evaluator import FloorPlanGenerator
from src.utils.config_parser import Config
from src.utils.utils import set_gpu_growth

set_gpu_growth()


def evaluate(config):

    config = Config(config)

    data_dir = config.get_string('data', './datasets/tfrecords')
    width = config.get_int('width', 256)
    height = config.get_int('height', 256)
    lat_dim = config.get_int('latent_dimensions', 32)
    meta_dim = config.get_int('meta_dimensions', 14)
    gen_ckpt = config.get_string('gen_ckpt')
    corner_ckpt = config.get_string('corner_ckpt')
    num_samples = config.get_int('num_samples', 3)

    floor_plan_dataset = FloorPlanDataset(None, data_type=FloorPlanDataType.TFRECORD)

    dataset = floor_plan_dataset.generate_dataset('test', max_samples=num_samples)

    evaluator = FloorPlanGenerator(dataset, lat_dim, meta_dim, gen_ckpt, corner_ckpt)

    dataset = dataset.batch(1).take(1)

    for index, (wa, d, wi, e, r, c, s, rt, wc, dc, wic, cl, ht) in dataset.enumerate():

        meta_input = tf.concat([rt, dc, wic], axis=1)
        evaluator.reconstruct(wa, d, wi, r, c, s, rt, meta_input, str(index),
                              save=True, show_walls=True, show_doors=True,
                              show_windows=True, show_rooms=True, show_corners=True,
                              show_shape=True, show_reconstructed=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to the config file. Default file: config/evaluate.yaml',
                        default='./config/evaluate.yaml')
    args = parser.parse_args()
    evaluate(args.config)
