import datetime
import os
import time
import sys
import cv2

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
from skimage import measure


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


def find_axis(pt1, pt2, threshold=1):

    if abs(pt1[0] - pt2[0]) <= threshold:
        return 0
    elif abs(pt1[1] - pt2[1]) <= threshold:
        return 1
    else:
        return -1


def p2p_distance(point_1, point_2):

    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))


def l2l_distance(line_1, line_2):

    line_1_direction = calc_line_direction(line_1)
    line_2_direction = calc_line_direction(line_2)
    assert(line_1_direction == line_2_direction)

    if line_1_direction == 0:
        return abs(line_1[0][1] - line_2[0][1])
    else:
        return abs(line_1[0][0] - line_2[0][0])


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


def float_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def set_gpu_growth():

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_harris_corners_img(gray_img):

    img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 1, 255, -1)

    return img


def get_hough_lines_img(wall):

    generated_img = np.zeros_like(wall, np.float32)
    hough_lines = cv2.HoughLinesP(np.uint8(wall), 1, np.pi/180, 5, 3)

    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(generated_img, (x1, y1), (x2, y2), color=255.0, thickness=1)

    return generated_img


def extract_corners(heatmaps, heatmap_threshold=0.5, pixel_threshold=3):

    heatmaps = (heatmaps > heatmap_threshold).astype(np.float32)
    orientation_points = []
    kernel = np.ones((3, 3), np.uint8)
    for heatmap_index in range(0, heatmaps.shape[0]):

        heatmap = heatmaps[heatmap_index, :, :]
        # heatmap = cv2.erode(heatmap, kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min() + 1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            if ys.shape[0] <= pixel_threshold:
                continue

            points.append((xs.mean(), ys.mean()))
        orientation_points.append(points)

    return orientation_points
