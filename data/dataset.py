import fnmatch
import os
from enum import Enum

import h5py
import tensorflow as tf


class FloorPlanDataType(Enum):
    TFRECORD = '*.tfrecord'
    HDF5 = '*.h5py'


class FloorPlanDataset:

    def __init__(self, data_dir, width, height, data_type=FloorPlanDataType.TFRECORD, num_parallel_reads=2):

        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.data_type = data_type
        self.num_parallel_reads = num_parallel_reads

    def generate_train_dataset(self, train_data_dir='train', max_samples=-1,
                               include_walls=True, include_doors=True,
                               include_windows=True, include_rooms=True,
                               include_shape=True, include_corners=False):

        if not (include_walls or include_doors or include_windows or
                include_rooms or include_shape or include_corners):
            raise AttributeError('At least one element should be included')

        if self.data_type == FloorPlanDataType.TFRECORD:
            return self._get_dataset_from_tfrecord(os.path.join(self.data_dir, train_data_dir),
                                                   max_samples, include_walls,
                                                   include_doors, include_windows, include_rooms,
                                                   include_shape, include_corners)

        elif self.data_type == FloorPlanDataType.HDF5:
            return self._get_dataset_from_h5(os.path.join(self.data_dir, train_data_dir),
                                             max_samples, include_walls,
                                             include_doors, include_windows, include_rooms,
                                             include_shape, include_corners)

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

    def _get_dataset_from_h5(self, data_dir, max_samples, include_walls, include_doors,
                             include_windows, include_rooms, include_shape, include_corners):

        """
        Generates a tensorflow dataset from hdf5 files which contain walls, doors, windows, rooms, shape and corners

        :param data_dir: Path to directory where the hdf5 files are stored
        :param max_samples: Maximum number of samples to be loaded
        :param include_walls: True if the final dataset should contain walls
        :param include_doors: True if the final dataset should contain doors
        :param include_windows: True if the final dataset should contain windows
        :param include_rooms: True if the final dataset should contain rooms
        :param include_shape: True if the final dataset should contain shape
        :param include_corners: True if the final dataset should contain corners
        :return: Tensorflow dataset
        """
        #TODO: This method is broken as of now, fix it

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
            lambda data: self._transform_and_filter_masks(
                data, include_walls, include_doors, include_windows,
                include_rooms, include_shape, include_corners),
            num_parallel_calls=self.num_parallel_reads)

        return dataset

    def _get_dataset_from_tfrecord(self, data_dir, max_samples, include_walls, include_doors,
                                   include_windows, include_rooms, include_shape, include_corners):

        file_names = self._get_all_file_names(data_dir, FloorPlanDataType.TFRECORD.value, max_samples)

        files = tf.data.Dataset.list_files(file_names)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
                                   cycle_length=self.num_parallel_reads,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(self._parse_function, num_parallel_calls=self.num_parallel_reads)

        dataset = dataset.map(
            lambda data: self._transform_and_filter_masks(
                data, include_walls, include_doors, include_windows,
                include_rooms, include_shape, include_corners),
            num_parallel_calls=self.num_parallel_reads)

        return dataset

    def _parse_function(self, example_proto):

        feature_description = {
            'wall_mask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'door_mask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'window_mask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'room_mask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'corner_mask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'bounding_mask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }

        return tf.io.parse_single_example(example_proto, feature_description)

    def _transform_and_filter_masks(self, data, include_walls, include_doors,
                                    include_windows, include_rooms, include_shape, include_corners):

        out = {}
        if include_walls:
            out['wall_mask'] = tf.reshape(data['wall_mask'], [self.width, self.height, 1])
        if include_doors:
            out['door_mask'] = tf.reshape(data['door_mask'], [self.width, self.height, 1])
        if include_windows:
            out['window_mask'] = tf.reshape(data['window_mask'], [self.width, self.height, 1])
        if include_rooms:
            out['room_mask'] = tf.reshape(data['room_mask'], [self.width, self.height, 10])
        if include_shape:
            out['shape_mask'] = tf.reshape(data['bounding_mask'], [self.width, self.height, 1])
        if include_corners:
            out['corner_mask'] = tf.reshape(data['corner_mask'], [self.width, self.height, 17])

        return out
