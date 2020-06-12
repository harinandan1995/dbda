import tensorflow as tf


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_floor_plan(string_dict: dict, float_dict: dict):

    feature = {}

    for k, v in string_dict.items():
        feature[k] = bytes_feature(tf.io.serialize_tensor(v))

    for k, v in float_dict.items():
        feature[k] = float_feature(v)

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def deserialize_floor_plan(example_proto):

    feature_description = {
        'wall_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'door_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'window_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'entrance_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'room_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'corner_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'shape_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'room_types': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'wall_count': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'door_count': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'window_count': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'cooling': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'heating': tf.io.FixedLenFeature([], tf.float32, default_value=0),
    }

    return tf.io.parse_single_example(example_proto, feature_description)

