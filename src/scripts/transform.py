from argparse import ArgumentParser

from src.data.data_transformer import VectorToImageTransformer, TransformerConfig
from src.utils.config import ConfigParser
from src.utils.utils import set_gpu_growth


def transform_vectors(config_path, params):

    config = ConfigParser(config_path, params).config

    transformer_config = TransformerConfig(wall_thickness=config.thickness.wall,
                                           window_thickness=config.thickness.window,
                                           door_thickness=config.thickness.door,
                                           corner_thickness=config.thickness.corner,
                                           inp_dir=config.dir.input,
                                           out_width=config.out.width,
                                           out_height=config.out.height,
                                           out_format=config.out.type,
                                           out_dir=config.dir.output)

    vector_to_image_transformer = VectorToImageTransformer(transformer_config)
    vector_to_image_transformer.transform_vectors_to_images(num_images=config.out.num_samples,
                                                            shuffle_data=config.shuffle)


if __name__ == '__main__':
    set_gpu_growth()

    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to the config file. Default file: config/transform.yaml',
                        default='config/transform.yaml')
    args = parser.parse_args()
    transform_vectors(args.config, None)
