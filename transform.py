from data.data_transformer import VectorToImageTransformer, TransformerConfig

config = TransformerConfig(wall_thickness=2,
                           inp_dir='./datasets/vectors',
                           out_width=256,
                           out_height=256,
                           out_format='tfrecord',
                           out_dir='/home/harikatam/TUM/sose2019/IDP/datasets/dummy')

vector_to_image_transformer = VectorToImageTransformer(config, num_images=1)
vector_to_image_transformer.transform_vectors_to_images()
