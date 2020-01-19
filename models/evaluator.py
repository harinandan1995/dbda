import math

import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from models.corner_detector import CornerDetector
from models.generator import Generator
from models.losses import reconstruction_loss_v2
from utils.utils import *


class FloorPlanGenerator:
    """
    For a given set of inputs (shape mask etc.) generates masks for walls, windows, doors,
    rooms, corners using the generator from the checkpoint

    :param dataset: TFDataset from where the shape masks are stored
    :param gen_ckpt_path: Path to the checkpoint from where the generator weights are loaded
    :param
    """

    def __init__(self, dataset, latent_dim, gen_ckpt_path, corner_ckpt_path,
                 out_dir='./out', width=256, height=256):

        self.dataset = dataset
        self.width = width
        self.height = height
        self.out_dir = os.path.join(out_dir, get_timestamp())
        self.latent_dim = latent_dim

        self.threshold = 2
        self.colors = np.random.randint(255, size=(20, 3))

        create_directory_if_not_exist(self.out_dir)

        self.latent_optimizer = tf.keras.optimizers.Adam(6e-2)
        self.generator = Generator(1, [3, 10], self.latent_dim, 12, gen_ckpt_path, width, height)
        self.corner_detector = CornerDetector(3, 17, corner_ckpt_path)

    def evaluate(self, max_samples=10):

        dataset = self.dataset.batch(1)

        # Walls, doors, windows, rooms, corners, shape, room types, wall count,
        # door count, window count
        for index, (wa, d, wi, e, r, c, s, rt, wc, dc, wic) in dataset.enumerate():

            if index >= max_samples > 0:
                return

            self._plot_original_data(wa, d, wi, r, c, s, index, save_output=True)

            self._plot_generated_data(wa, d, wi, r, c, s, rt, wc, dc, wic, index, save_output=True)

    def _plot_original_data(self, wa, d, wi, r, c, s, index, save_output=False):
        """
        Plots the masks of the original data

        :param wa: Wall mask
        :param d: Door mask
        :param wi: Window mask
        :param r: Room masks
        :param c: Corner masks
        :param s: Shape mask
        :param save_output: True or False. When True a copy of the plot as an image is stored in
        the output directory
        """
        rows = 6
        columns = 6
        fig = plt.figure(figsize=(20, 12))
        fig.add_subplot(rows, columns, 1)
        wall_mask = np.rollaxis(wa[0].numpy(), 2, 0)[0]
        plt.imshow(wall_mask, cmap='hot', interpolation='nearest')
        fig.add_subplot(rows, columns, 2)
        door_mask = np.rollaxis(d[0].numpy(), 2, 0)[0]
        plt.imshow(door_mask, cmap='hot', interpolation='nearest')
        fig.add_subplot(rows, columns, 3)
        window_mask = np.rollaxis(wi[0].numpy(), 2, 0)[0]
        plt.imshow(window_mask, cmap='hot', interpolation='nearest')

        for i in range(10):
            target_numpy = np.rollaxis(r[0].numpy(), 2, 0)
            fig.add_subplot(rows, columns, i + 4)
            plt.imshow(target_numpy[i], cmap='hot', interpolation='nearest')

        for i in range(17):
            target_numpy = np.rollaxis(c[0].numpy(), 2, 0)
            fig.add_subplot(rows, columns, i + 14)
            plt.imshow(target_numpy[i], cmap='hot', interpolation='nearest')

        fig.add_subplot(rows, columns, 31)
        shape_mask = np.rollaxis(s[0].numpy(), 2, 0)[0]
        plt.imshow(shape_mask, cmap='hot', interpolation='nearest')

        if save_output:
            plt.savefig(os.path.join(self.out_dir, '%d_original.png' % index))

        plt.show()

    def _plot_generated_data(self, wa, d, wi, r, c, s, rt, wc, dc, wic, index, save_output=True):

        """
        Plots the heatmap of the generated data for the given shape of the building

        :param s: The shape mask for the generator
        :param save_output: True or False. When True a copy of the plot as an image is stored in
        the output directory
        """

        # wdw -> walls, doors, windows
        # Get the output of the generator
        meta_input = tf.concat([rt, dc, wic], axis=1)
        wdw_target = tf.concat([wa, d, wi], axis=3)
        room_target = r

        latent_variable = tf.Variable(tf.random.normal([meta_input.shape[0], self.latent_dim]),
                                      trainable=True, validate_shape=True)

        wdw_gen_out, room_gen_out, latent_loss = self._train_latent_variable(
            latent_variable, s, meta_input, wdw_target, room_target, rt, 600)
        corner_input = tf.cast(tf.greater(wdw_gen_out, 0.7), tf.float32)

        wdw_gen_out = np.rollaxis(wdw_gen_out.numpy()[0], 2, 0)

        fig = plt.figure(figsize=(12, 12))
        rows = 6
        columns = 6
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self._filter_heatmap(wdw_gen_out[0]), cmap='hot', interpolation='nearest')
        fig.add_subplot(rows, columns, 2)
        plt.imshow(self._filter_heatmap(wdw_gen_out[1]), cmap='hot', interpolation='nearest')
        fig.add_subplot(rows, columns, 3)
        plt.imshow(self._filter_heatmap(wdw_gen_out[2]), cmap='hot', interpolation='nearest')

        corner_out = self.corner_detector(corner_input, training=False)
        corner_out = np.rollaxis(corner_out.numpy()[0], 2, 0)
        corner_points = extract_corners(corner_out, 0.5, 3)
        corner_points = self._cluster_corner_points(corner_points)

        room_gen_out = np.rollaxis(room_gen_out.numpy()[0], 2, 0)
        for i in range(10):
            fig.add_subplot(rows, columns, i + 4)
            plt.imshow(self._filter_heatmap(room_gen_out[i]), cmap='hot', interpolation='nearest')

        for i in range(17):
            fig.add_subplot(rows, columns, i + 14)
            plt.imshow(self._filter_heatmap(corner_out[i]), cmap='hot', interpolation='nearest')

        wall_mask, door_mask, window_mask = self._generate_text_file(
            corner_points, self._filter_heatmap(wdw_gen_out[1]), self._filter_heatmap(wdw_gen_out[2]),
            np.rollaxis(s.numpy()[0], 2, 0), self._filter_heatmap(room_gen_out),
            os.path.join(self.out_dir, "out_%d.txt" % index)
        )

        fig.add_subplot(rows, columns, 31)
        plt.imshow(wall_mask, cmap='hot', interpolation='nearest')
        fig.add_subplot(rows, columns, 32)
        plt.imshow(door_mask, cmap='hot', interpolation='nearest')
        fig.add_subplot(rows, columns, 33)
        plt.imshow(window_mask, cmap='hot', interpolation='nearest')

        if save_output:
            plt.savefig(os.path.join(self.out_dir, '%d_generated.png' % index))
        plt.show()

    def _train_latent_variable(self, latent_variable, shape, meta_input, wdw_target,
                               room_target, room_type, iterations):

        latent_loss = 100

        for lat_step in range(iterations):

            with tf.GradientTape() as latent_tape:

                wdw_gen_out, room_gen_out = self.generator([shape, meta_input, latent_variable], training=False)
                latent_loss = reconstruction_loss_v2(shape, wdw_gen_out, room_gen_out,
                                                     wdw_target, room_target, room_type)

            latent_gradients = latent_tape.gradient(latent_loss, latent_variable)
            self.latent_optimizer.apply_gradients(zip([latent_gradients], [latent_variable]))

            latent_norm = tf.norm(latent_variable, axis=1, keepdims=True)
            latent_norm = tf.tile(latent_norm, [1, self.latent_dim])
            latent_variable.assign(tf.divide(latent_variable, latent_norm) * math.sqrt(self.latent_dim))

            if lat_step % 5 == 0 or lat_step + 1 == iterations:
                print('Step %d Lat loss %f' % (lat_step + 1, latent_loss))

        return wdw_gen_out, room_gen_out, latent_loss

    def _generate_text_file(self, corner_points, door_heatmap, window_heatmap, shape, room_heatmap, file_path):

        walls = self._get_walls(corner_points)
        doors, windows = self._get_doors_and_windows(corner_points, door_heatmap, window_heatmap)

        wall_mask = np.zeros((self.width, self.height), dtype=np.int8)
        for pt1, pt2 in walls:
            if pt2 is not None:
                cv2.line(wall_mask, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=1, thickness=3)

        door_mask = np.zeros((self.width, self.height), dtype=np.int8)
        for pt1, pt2 in doors:
            if pt2 is not None:
                cv2.line(door_mask, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=1, thickness=3)

        window_mask = np.zeros((self.width, self.height), dtype=np.int8)
        for pt1, pt2 in windows:
            if pt2 is not None:
                cv2.line(window_mask, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=1, thickness=3)

        room_centers = self._get_room_centers(wall_mask, shape, room_heatmap)

        text_file = open(file_path, 'w')

        for wall in walls:
            if wall[1] is not None:
                text_file.write("%d %d %d %d wall\n" % (wall[0][0], wall[0][1], wall[1][0], wall[1][1]))
        for door in doors:
            if door[1] is not None:
                text_file.write("%d %d %d %d door\n" % (door[0][0], door[0][1], door[1][0], door[1][1]))
        for window in windows:
            if window[1] is not None:
                text_file.write("%d %d %d %d window\n" % (window[0][0], window[0][1], window[1][0], window[1][1]))
        for room_center in room_centers:
            text_file.write("%d %d room_%d\n" % (room_center[0][1], room_center[0][1], room_center[1]))

        text_file.close()

        return wall_mask, door_mask, window_mask

    def _get_walls(self, corner_points):

        lines = []
        top_points = self._append_point(corner_points, [2, 5, 6, 9, 10, 11, 12])
        right_points = self._append_point(corner_points, [1, 5, 4, 8, 10, 9, 12])
        bottom_points = self._append_point(corner_points, [0, 7, 4, 8, 9, 11, 12])
        left_points = self._append_point(corner_points, [3, 7, 6, 8, 11, 10, 12])

        for i in range(13):

            for (x, y) in corner_points[i]:

                if i == 0 or i == 7 or i == 4 or i == 8 or i == 11 or i == 9 or i == 12:
                    lines.append(self._find_closest_top_point(x, y, top_points))
                if i == 3 or i == 7 or i == 6 or i == 8 or i == 11 or i == 10 or i == 12:
                    lines.append(self._find_closest_right_point(x, y, right_points))
                if i == 2 or i == 6 or i == 5 or i == 9 or i == 10 or i == 11 or i == 12:
                    lines.append(self._find_closest_bottom_point(x, y, bottom_points))
                if i == 1 or i == 4 or i == 5 or i == 8 or i == 9 or i == 10 or i == 12:
                    lines.append(self._find_closest_left_point(x, y, left_points))

        self._filter_intersecting_lines(lines)

        return lines

    def _get_doors_and_windows(self, corner_points, door_heatmap, window_heatmap):

        lines = []
        top_points = self._append_point(corner_points, [16])
        right_points = self._append_point(corner_points, [15])
        bottom_points = self._append_point(corner_points, [14])
        left_points = self._append_point(corner_points, [13])

        for i in range(13, 17, 1):

            for (x, y) in corner_points[i]:

                if i == 13:
                    lines.append(self._find_closest_right_point(x, y, right_points))
                if i == 16:
                    lines.append(self._find_closest_bottom_point(x, y, bottom_points))
                if i == 15:
                    lines.append(self._find_closest_left_point(x, y, left_points))
                if i == 14:
                    lines.append(self._find_closest_top_point(x, y, top_points))

        windows = []
        doors = []

        for line in lines:
            if line[1] is None:
                continue
            min_x = int(min(line[0][0], line[1][0]))
            min_y = int(min(line[0][1], line[1][1]))
            max_x = int(max(line[0][0], line[1][0]))
            max_y = int(max(line[0][1], line[1][1]))
            if np.mean(door_heatmap[min_y-1:max_y+1, min_x-1:max_x+1]) > 0.1:
                doors.append(line)
            elif np.mean(window_heatmap[min_y-1:max_y+1, min_x-1:max_x+1]) > 0.1:
                windows.append(line)

        self._filter_intersecting_lines(windows)
        self._filter_intersecting_lines(doors)

        return doors, windows

    def _get_room_centers(self, walls, shape, rooms):

        final_mask = walls + shape[0]
        room_segmentation = measure.label(final_mask == 1, background=0)

        room_centers = []

        for i in range(np.amax(room_segmentation)+1):
            if i == 0:
                continue

            for j in range(10):
                if (np.mean(rooms[j][room_segmentation == i]) > 0.3):
                    x, y = np.where(room_segmentation == i)
                    mean_x = 0
                    mean_y = 0
                    for x1, y1 in zip(x, y):
                        mean_x += x1
                        mean_y += y1

                    mean_x /= len(x)
                    mean_y /= len(y)
                    room_centers.append(((int(mean_y), int(mean_x)), j))
                    break

        return room_centers

    def _cluster_corner_points(self, corner_points):

        all_points = []
        all_indices = []

        for i in range(len(corner_points)):
            for point in corner_points[i]:
                all_points.append(point)
                all_indices.append(i)

        Z = linkage(all_points, method='complete',  metric='euclidean')

        max_d = 4
        clusters = fcluster(Z, max_d, criterion='distance')

        cluster_sets = {}

        for i, cluster_id in enumerate(clusters):
            if str(cluster_id) in cluster_sets:
                cluster_sets[str(cluster_id)].append(all_points[i])
            else:
                cluster_sets[str(cluster_id)] = [all_points[i]]

        cluster_centroids = {}
        for cluster_id, points in cluster_sets.items():
            mean_x = 0
            mean_y = 0
            for x, y in points:
                mean_x += x
                mean_y += y

            mean_x /= len(points)
            mean_y /= len(points)
            cluster_centroids[cluster_id] = (int(mean_x), int(mean_y))

        cluster_points = []
        for i in range(len(corner_points)):
            cluster_points.append([])

        for i, cluster_id in enumerate(clusters):
            cluster_points[all_indices[i]].append(cluster_centroids[str(cluster_id)])

        return cluster_points

    @staticmethod
    def _filter_intersecting_lines(lines):

        invalid_line_indices = {}

        for i, line_1 in enumerate(lines):
            if line_1[1] is None:
                invalid_line_indices[i] = True
                continue
            for j, line_2 in enumerate(lines):
                if line_2[1] is None:
                    invalid_line_indices[j] = True
                    continue

                if i in invalid_line_indices or j in invalid_line_indices:
                    continue
                if check_if_walls_touch(line_1, line_2, 2) and i < j:
                    lines[i] = merge_lines(line_1, line_2)
                    invalid_line_indices[j] = True

        return [line for index, line in enumerate(lines) if index not in invalid_line_indices]

    @staticmethod
    def _append_point(points, indices):

        out_list = []

        for i in indices:
            out_list += points[i]

        return out_list

    def _find_closest_top_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if y + self.threshold >= pt_y >= y - self.threshold:
                if x - pt_x > 1 and abs(x - pt_x) < distance:
                    distance = abs(x - pt_x)
                    closest_point = (pt_x, (pt_y + y)/2)

        return (x, y), closest_point

    def _find_closest_bottom_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if y + self.threshold >= pt_y >= y - self.threshold:
                if pt_x - x > 1 and abs(x - pt_x) < distance:
                    distance = abs(x - pt_x)
                    closest_point = (pt_x, (pt_y + y)/2)

        return (x, y), closest_point

    def _find_closest_left_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if x + self.threshold >= pt_x >= x - self.threshold:
                if y - pt_y > 1 and abs(y - pt_y) < distance:
                    distance = abs(y - pt_y)
                    closest_point = ((pt_x + x)/2, pt_y)

        return (x, y), closest_point

    def _find_closest_right_point(self, x, y, points):

        distance = 100000
        closest_point = None
        for (pt_x, pt_y) in points:
            if x + self.threshold >= pt_x >= x - self.threshold:
                if pt_y - y > 1 and abs(y - pt_y) < distance:
                    distance = abs(y - pt_y)
                    closest_point = ((pt_x + x)/2, pt_y)

        return (x, y), closest_point

    @staticmethod
    def _filter_heatmap(heatmap, threshold=0.7):

        filtered_heatmap = heatmap
        filtered_heatmap[heatmap <= threshold] = 0
        filtered_heatmap[heatmap > threshold] = 1

        return filtered_heatmap

    @staticmethod
    def _apply_nms(heatmap, window_size=3):

        return heatmap * (heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))

    @staticmethod
    def _mask_to_segmentation_image(mask):
        """
        Converts a mask image with dimensions HxWxC into a segmentation image with C labels

        :param mask: A HXWXC mask image
        :return: A segmentation image with C labels
        """
        num_colors = mask.shape[0]

        segmentation_image = 255 * np.ones((mask.shape[1], mask.shape[2], 3), np.uint8)

        for i in range(mask.shape[0]):
            segmentation_image[mask[i] == 1] = self.colors[i]

        return segmentation_image
