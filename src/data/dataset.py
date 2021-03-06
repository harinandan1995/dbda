import fnmatch
import os
from enum import Enum

import h5py
import tensorflow as tf

from src.utils.config import ConfigParser
from src.nnutils.tfrecord import deserialize_floor_plan


class FloorPlanDataType(Enum):
    TFRECORD = '*.tfrecord'
    HDF5 = '*.h5py'


class FloorPlanDataset:

    def __init__(self, config: ConfigParser.ConfigObject,
                 data_type=FloorPlanDataType.TFRECORD, num_parallel_reads=4):

        self.config = config
        self.data_dir = config.data_dir
        self.data_type = data_type
        self.num_parallel_reads = num_parallel_reads

    def generate_dataset(self, train_data_dir='train', max_samples=-1):

        if self.data_type == FloorPlanDataType.TFRECORD:
            return self._get_dataset_from_tfrecord(os.path.join(self.data_dir, train_data_dir), max_samples)

        elif self.data_type == FloorPlanDataType.HDF5:
            return self._get_dataset_from_h5(os.path.join(self.data_dir, train_data_dir), max_samples)

        else:
            raise AttributeError('Invalid value provided for data type. Allowed values: tfrecord, h5')

    @staticmethod
    def _get_all_file_names(data_dir, file_extension, max_samples):

        num_samples = 0
        out_file_names = []
        for r, d, filenames in os.walk(os.path.join(data_dir)):
            for file in fnmatch.filter(filenames, file_extension):

                if num_samples >= max_samples > 0:
                    return out_file_names
                out_file_names.append(os.path.join(r, file))
                num_samples += 1

        return out_file_names

    def _get_dataset_from_h5(self, data_dir, max_samples):

        """
        Generates a tensorflow dataset from hdf5 files which contain walls, doors, windows, rooms, shape and corners

        :param data_dir: Path to directory where the hdf5 files are stored
        :param max_samples: Maximum number of samples to be loaded
        :return: Tensorflow dataset
        """
        # TODO: This method is broken as of now, fix it

        dataset = None

        file_names = self._get_all_file_names(data_dir, FloorPlanDataType.HDF5.value, max_samples)

        for file_name in file_names:
            f = h5py.File(file_name, 'r')

            print(file_name, f.keys())

            if dataset is None:
                dataset = tf.data.Dataset.from_tensors((f['wall_mask'], f['door_mask'], f['window_mask'],
                                                        f['room_mask'], f['bounding_mask'], f['corner_mask']))
            else:
                dataset = dataset.concatenate(
                    tf.data.Dataset.from_tensors((f['wall_mask'], f['door_mask'],
                                                  f['window_mask'], f['room_mask'],
                                                  f['bounding_mask'], f['corner_mask'])))

        dataset = dataset.map(
            lambda data: self._transform_and_filter_masks(data),
            num_parallel_calls=self.num_parallel_reads)

        return dataset

    def _get_dataset_from_tfrecord(self, data_dir, max_samples):

        file_names = self._get_all_file_names(data_dir, FloorPlanDataType.TFRECORD.value, max_samples)

        files = tf.data.Dataset.list_files(file_names)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
                                   cycle_length=self.num_parallel_reads,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(deserialize_floor_plan, num_parallel_calls=self.num_parallel_reads)

        dataset = dataset.map(
            lambda data: self._transform_and_filter_masks(data),
            num_parallel_calls=self.num_parallel_reads)

        return dataset

    @staticmethod
    def _transform_and_filter_masks(data):

        data['wall_mask'] = tf.io.parse_tensor(data['wall_mask'], tf.float32)
        data['door_mask'] = tf.io.parse_tensor(data['door_mask'], tf.float32)
        data['window_mask'] = tf.io.parse_tensor(data['window_mask'], tf.float32)
        data['entrance_mask'] = tf.io.parse_tensor(data['entrance_mask'], tf.float32)
        data['room_mask'] = tf.io.parse_tensor(data['room_mask'], tf.float32)
        data['corner_mask'] = tf.io.parse_tensor(data['corner_mask'], tf.float32)
        data['shape_mask'] = tf.io.parse_tensor(data['shape_mask'], tf.float32)
        data['room_types'] = tf.io.parse_tensor(data['room_types'], tf.float32)

        data['wall_count'] = tf.cast(data['wall_count'], tf.float32)
        data['door_count'] = tf.cast(data['door_count'], tf.float32)
        data['window_count'] = tf.cast(data['window_count'], tf.float32)
        data['cooling'] = tf.cast(data['cooling'], tf.float32)
        data['heating'] = tf.cast(data['heating'], tf.float32)

        return data
