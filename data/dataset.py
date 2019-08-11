import fnmatch
import os
from enum import Enum

import h5py

import tensorflow as tf


class FloorPlanDataType(Enum):
    TFRECORD = 'tfrecord'
    HDF5 = 'h5'


class FloorPlanDataset:

    def __init__(self, data_dir, width, height, data_type=FloorPlanDataType.TFRECORD, num_parallel_reads=2):

        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.data_type = data_type
        self.num_parallel_reads = num_parallel_reads

    def generate_train_dataset(self, train_data_dir='train', include_walls=True, include_doors=True,
                               include_windows=True, include_rooms=True,
                               include_shape=True, include_corners=False):

        if self.data_type.value == 'tfrecord':
            return self._get_dataset_from_tfrecord(os.path.join(self.data_dir, train_data_dir), include_walls,
                                                   include_doors, include_windows, include_rooms,
                                                   include_shape, include_corners)

        elif self.data_type.value == 'h5':
            return self._get_dataset_from_h5(os.path.join(self.data_dir, train_data_dir), include_walls,
                                             include_doors, include_windows, include_rooms,
                                             include_shape, include_corners)

        else:
            raise AttributeError('Invalid value provided for data type. Allowed values: tfrecord, h5')

    def _get_dataset_from_h5(self, data_dir, include_walls, include_doors, include_windows,
                             include_rooms, include_shape, include_corners):

        dataset = None

        for r, d, filenames in os.walk(os.path.join(data_dir)):
            for filename in fnmatch.filter(filenames, '*.h5py'):

                f = h5py.File(os.path.join(r, filename), 'r')

                print(filename, f.keys())

                if dataset is None:
                    dataset = tf.data.Dataset.from_tensors((f['wall_mask'], f['door_mask'], f['window_mask'],
                                                            f['room_mask'], f['bounding_mask'], f['corner_mask']))
                else:
                    dataset = dataset.concatenate(
                        tf.data.Dataset.from_tensors((f['wall_mask'], f['door_mask'],
                                                      f['window_mask'], f['room_mask'],
                                                      f['bounding_mask'], f['corner_mask'])))

        dataset = dataset.map(
            lambda wa, do, wi, ro, sh, co: self._transform_and_filter_masks(wa, do, wi, ro, sh, co,
                                                                            include_walls, include_doors,
                                                                            include_windows,
                                                                            include_rooms, include_shape,
                                                                            include_corners),
            num_parallel_calls=self.num_parallel_reads)

        return dataset

    def _get_dataset_from_tfrecord(self, data_dir, include_walls, include_doors, include_windows,
                                   include_rooms, include_shape, include_corners):

        feature_description = {
            'wall_mask': tf.FixedLenFeature([], tf.float32),
            'door_mask': tf.FixedLenFeature([], tf.float32),
            'window_mask': tf.FixedLenFeature([], tf.float32),
            'room_mask': tf.FixedLenFeature([], tf.float32),
            'bounding_mask': tf.FixedLenFeature([], tf.float32),
            'corner_mask': tf.FixedLenFeature([], tf.float32)
        }

        file_names = []
        for r, d, filenames in os.walk(os.path.join(data_dir)):
            for file in fnmatch.filter(filenames, '*.tfrecord'):
                filenames.append(os.path.join(r, file))

        files = tf.data.Dataset.list_files(file_names)
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=self.num_parallel_reads,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(lambda x: self._parse_function(x, feature_description),
                              num_parallel_calls=self.num_parallel_reads)

        dataset = dataset.map(
            lambda wa, do, wi, ro, sh, co: self._transform_and_filter_masks(wa, do, wi, ro, sh, co,
                                                                            include_walls, include_doors,
                                                                            include_windows,
                                                                            include_rooms, include_shape,
                                                                            include_corners),
            num_parallel_calls=self.num_parallel_reads)

        return dataset

    @staticmethod
    def _parse_function(example_proto, feature_description):

        return tf.parse_single_example(example_proto, feature_description)

    def _transform_and_filter_masks(self, wa, do, wi, ro, sh, co, include_walls, include_doors,
                                    include_windows, include_rooms, include_shape, include_corners):

        out = ()
        if include_walls:
            out = out + tf.reshape(wa, [self.width, self.height, 1])
        if include_doors:
            out = out + tf.reshape(do, [self.width, self.height, 1])
        if include_windows:
            out = out + tf.reshape(wi, [self.width, self.height, 1])
        if include_rooms:
            out = out + tf.reshape(ro, [self.width, self.height, 11])
        if include_shape:
            out = out + tf.reshape(sh, [self.width, self.height, 1])
        if include_corners:
            out = out + tf.reshape(co, [self.width, self.height, 13])

        return out
