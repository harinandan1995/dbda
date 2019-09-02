import datetime
import os
import time

import numpy as np
import tensorflow as tf


# Create a directories in the path if they dont exist
def create_directory_if_not_exist(path):

    if not os.path.exists(path):
        os.makedirs(path)


def get_day():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d')


def get_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%H%M%S')


def get_timestamp():

    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')


def p2p_distance(point_1, point_2):

    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))


def calc_line_direction(line):
    return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))


def is_manhattan(line, gap=3):
    return min(abs(line[0][0] - line[1][0]), abs(line[0][1] - line[1][1])) < gap


def transform_point(transformation, point):
    point = np.array(point)
    point = np.concatenate([point, np.ones(1)], axis=0)
    point = np.matmul(transformation, point)
    return tuple(np.round(point[:2] / point[2]).astype(np.int32).tolist())


def line_range(line):

    direction = calc_line_direction(line)
    fixed_value = (line[0][1 - direction] + line[1][1 - direction]) // 2
    min_value = min(line[0][direction], line[1][direction])
    max_value = max(line[0][direction], line[1][direction])

    return direction, fixed_value, min_value, max_value


def merge_lines(line_1, line_2):

    direction_1, fixed_value_1, min_1, max_1 = line_range(line_1)
    direction_2, fixed_value_2, min_2, max_2 = line_range(line_2)
    fixed_value = (fixed_value_1 + fixed_value_2) // 2

    if direction_1 == 0:
        return [(min(min_1, min_2), fixed_value), (max(max_1, max_2), fixed_value)]
    else:
        return [(fixed_value, min(min_1, min_2)), (fixed_value, max(max_1, max_2))]


def convert_to_point(x, y):

    return int(round(float(x))), int(round(float(y)))


def variable_summaries(var, step, prefix):

    mean = tf.reduce_mean(var)
    tf.summary.scalar(prefix + '_mean', mean, step)

    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.summary.scalar(prefix + '_stddev', stddev, step)
    tf.summary.scalar(prefix + '_max', tf.reduce_max(var), step)
    tf.summary.scalar(prefix + '_min', tf.reduce_min(var), step)
    tf.summary.histogram(prefix + '_histogram', var, step)
