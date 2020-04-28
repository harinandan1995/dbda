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

    def __init__(self, dataset, latent_dim, meta_dim, gen_ckpt_path, corner_ckpt_path,
                 out_dir='./out', width=256, height=256):

        self.dataset = dataset
        self.width = width
        self.height = height
        self.out_dir = os.path.join(out_dir, get_timestamp())
        self.latent_dim = latent_dim

        self.threshold = 2
        self.colors = np.random.randint(255, size=(20, 3))

        create_directory_if_not_exist(self.out_dir)

        self.latent_optimizer = tf.keras.optimizers.Adam(2e-2)
        self.iterations = 500
        self.generator = Generator(1, [3, 10], self.latent_dim, meta_dim,
                                   gen_ckpt_path, width, height)
        self.corner_detector = CornerDetector(3, 17, corner_ckpt_path)

        self.room_map = {
            0: 'living_room', 1: 'kitchen', 2: 'bedroom',
            3: 'bathroom', 5: 'closet', 6: 'balcony',
            7: 'corridor', 8: 'dining_room', 
            9: 'laundry_room'
        }

    def get_possible_plans(self, shape, meta_input, number_of_outputs=10, show_walls=True,
                           show_doors=True, show_windows=True, show_rooms=True, show_corners=True,
                           show_shape=True, show_reconstructed=True, save=True):

        if number_of_outputs < 1:
            return

        for i in range(number_of_outputs):

            latent_variable = tf.Variable(
                tf.random.normal([meta_input.shape[0], self.latent_dim]), validate_shape=True)

            wdw_gen_out, room_gen_out = self.generator([shape, meta_input, latent_variable], training=False)
            out_file_name = os.path.join(self.out_dir, '%d_original' % i)
            self._get_figure(shape, wdw_gen_out, room_gen_out, out_file_name, show_walls, show_doors,
                             show_windows, show_rooms, show_corners, show_shape, show_reconstructed, save)

    def reconstruct(self, wa, d, wi, r, c, s, rt, meta_input,
                    prefix, save=True, show_walls=True, show_doors=True,
                    show_windows=True, show_rooms=True, show_corners=True,
                    show_shape=True, show_reconstructed=True):

        self._plot_original_data(wa, d, wi, r, c, s, prefix,
                                 show_walls, show_doors, show_windows,
                                 show_rooms, show_corners, show_shape, save)
        self._plot_reconstructed_data(wa, d, wi, r, s, rt, meta_input,
                                      prefix, show_walls, show_doors, show_windows,
                                      show_rooms, show_corners, show_shape, show_reconstructed, save)

    @staticmethod
    def _get_figure_size(show_walls, show_doors, show_windows, show_rooms,
                         show_corners, show_shape, show_reconstructed):

        total_plots = 0.0
        if show_walls:
            total_plots += 1.0
        if show_doors:
            total_plots += 1.0
        if show_windows:
            total_plots += 1.0
        if show_rooms:
            total_plots += 10.0
        if show_corners:
            total_plots += 17.0
        if show_shape:
            total_plots += 1.0
        if show_reconstructed:
            total_plots += 3.0

        rows = int(math.sqrt(total_plots))
        columns = rows
        if total_plots / rows > rows:
            columns = rows + 1
            rows += 1

        return rows, columns

    def _get_figure(self, shape, wdw_gen_out, room_gen_out, out_file_name, show_walls, show_doors,
                    show_windows, show_rooms, show_corners, show_shape, show_reconstructed, save):

        index = 1
        rows, columns = self._get_figure_size(show_walls, show_doors, show_windows, show_rooms,
                                              show_corners, show_shape, show_reconstructed)
        fig = plt.figure(figsize=(rows * 2, columns * 2))

        wdw_gen_out_np = np.rollaxis(wdw_gen_out.numpy()[0], 2, 0)
        if show_walls:
            fig.add_subplot(rows, columns, index)
            plt.imshow(self._filter_heatmap(wdw_gen_out_np[0]), cmap='hot', interpolation='nearest')
            index += 1

        if show_doors:
            fig.add_subplot(rows, columns, index)
            plt.imshow(self._filter_heatmap(wdw_gen_out_np[1]), cmap='hot', interpolation='nearest')
            index += 1

        if show_windows:
            fig.add_subplot(rows, columns, index)
            plt.imshow(self._filter_heatmap(wdw_gen_out_np[2]), cmap='hot', interpolation='nearest')
            index += 1

        room_gen_out_np = np.rollaxis(room_gen_out.numpy()[0], 2, 0)
        if show_rooms:
            for i in range(10):
                fig.add_subplot(rows, columns, index)
                plt.imshow(self._filter_heatmap(room_gen_out_np[i]), cmap='hot', interpolation='nearest')
                index += 1

        corner_input = tf.cast(tf.greater(wdw_gen_out, 0.7), tf.float32)
        corner_out = self.corner_detector(corner_input, training=False)
        corner_out = np.rollaxis(corner_out.numpy()[0], 2, 0)
        corner_points = extract_corners(corner_out, 0.5, 3)
        corner_points = self._cluster_corner_points(corner_points)
        if show_corners:
            for i in range(17):
                fig.add_subplot(rows, columns, index)
                plt.imshow(self._filter_heatmap(corner_out[i]), cmap='hot', interpolation='nearest')
                index += 1

        wall_mask, door_mask, window_mask = self._generate_text_file(
            corner_points,
            self._filter_heatmap(wdw_gen_out_np[1]),
            self._filter_heatmap(wdw_gen_out_np[2]),
            np.rollaxis(shape.numpy()[0], 2, 0),
            self._filter_heatmap(room_gen_out_np),
            os.path.join(out_file_name+'.txt')
        )

        if show_shape:
            fig.add_subplot(rows, columns, index)
            plt.imshow(shape[0].numpy(), cmap='hot', interpolation='nearest')
            index += 1

        if show_reconstructed:

            fig.add_subplot(rows, columns, index)
            plt.imshow(wall_mask, cmap='hot', interpolation='nearest')
            fig.add_subplot(rows, columns, index + 1)
            plt.imshow(door_mask, cmap='hot', interpolation='nearest')
            fig.add_subplot(rows, columns, index + 2)
            plt.imshow(window_mask, cmap='hot', interpolation='nearest')

        if save:
            plt.savefig(out_file_name+'.png')

        plt.show()

    def _plot_reconstructed_data(self, wa, d, wi, r, s, rt, meta_input, prefix,
                                 show_walls, show_doors, show_windows, show_rooms,
                                 show_corners, show_shape, show_reconstructed, save):

        wdw_target = tf.concat([wa, d, wi], axis=3)
        room_target = r

        latent_variable = tf.Variable(tf.random.normal([meta_input.shape[0], self.latent_dim]),
                                      trainable=True, validate_shape=True)

        wdw_gen_out, room_gen_out, latent_loss = self._train_latent_variable(
            latent_variable, s, meta_input, wdw_target, room_target, rt, self.iterations)

        out_file_name = os.path.join(self.out_dir, prefix+'_recon')
        self._get_figure(s, wdw_gen_out, room_gen_out, out_file_name, show_walls, show_doors,
                         show_windows, show_rooms, show_corners, show_shape, show_reconstructed, save)

    def _plot_original_data(self, walls, doors, windows, rooms, corners, shape, prefix,
                            show_walls, show_doors, show_windows, show_rooms,
                            show_corners, show_shape, save):

        out_file_name = os.path.join(self.out_dir, prefix + '_original')

        index = 1
        rows, columns = self._get_figure_size(show_walls, show_doors, show_windows, show_rooms,
                                              show_corners, show_shape, False)
        fig = plt.figure(figsize=(rows * 2, columns * 2))

        if show_walls:
            wall_mask = np.rollaxis(walls[0].numpy(), 2, 0)[0]
            fig.add_subplot(rows, columns, index)
            plt.imshow(wall_mask, cmap='hot', interpolation='nearest')
            index += 1

        if show_doors:
            door_mask = np.rollaxis(doors[0].numpy(), 2, 0)[0]
            fig.add_subplot(rows, columns, index)
            plt.imshow(door_mask, cmap='hot', interpolation='nearest')
            index += 1

        if show_windows:
            window_mask = np.rollaxis(windows[0].numpy(), 2, 0)[0]
            fig.add_subplot(rows, columns, index)
            plt.imshow(window_mask, cmap='hot', interpolation='nearest')
            index += 1

        if show_rooms:
            room_masks = np.rollaxis(rooms[0].numpy(), 2, 0)
            for i in range(10):
                fig.add_subplot(rows, columns, index)
                plt.imshow(room_masks[i], cmap='hot', interpolation='nearest')
                index += 1

        if show_corners:
            corner_masks = np.rollaxis(corners[0].numpy(), 2, 0)
            for i in range(17):
                fig.add_subplot(rows, columns, index)
                plt.imshow(corner_masks[i], cmap='hot', interpolation='nearest')
                index += 1

        if show_shape:
            shape_mask = np.rollaxis(shape[0].numpy(), 2, 0)[0]
            fig.add_subplot(rows, columns, index)
            plt.imshow(shape_mask, cmap='hot', interpolation='nearest')
            index += 1

        if save:
            plt.savefig(out_file_name + '.png')

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
            text_file.write("%d %d 0 0 %s\n" % (room_center[0][1], room_center[0][1], self.room_map[room_center[1]]))

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

    @staticmethod
    def _get_room_centers(walls, shape, rooms):

        final_mask = walls + shape[0]
        room_segmentation = measure.label(final_mask == 1, background=0)

        room_centers = []

        for i in range(np.amax(room_segmentation)+1):
            if i == 0:
                continue

            for j in range(10):
                if np.mean(rooms[j][room_segmentation == i]) > 0.3:

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

    @staticmethod
    def _cluster_corner_points(corner_points):

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
