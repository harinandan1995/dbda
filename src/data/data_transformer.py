import glob
import h5py

from tqdm import tqdm
from random import shuffle
from src.nnutils.tfrecord import serialize_floor_plan
from src.utils.utils import *


class TransformerConfig:

    def __init__(self, wall_thickness=3, window_thickness=2, door_thickness=2,
                 inp_dir='../datasets/vectors', out_dir='../datasets/tfrecords',
                 out_format='tfrecord', out_width=256, out_height=256,
                 room_map=None, icon_map=None, color_map=None, corner_thickness=2):

        """
        Config for the vector to mask/image transformation

        :param inp_dir: Directory where the vector files are stored
        :param out_dir: Directory where the output file are stored
        :param out_format: Format of the output file. Allowed values: tfrecord, h5, png
        :param out_width: Width of the output mask/image
        :param out_height: Height of the output mask/image
        :param wall_thickness: Integer value for the thickness of the wall in the mask/image
        :param room_map: Map with name of room as key and label as value. If None loads a default map
        :param icon_map: Map with name of icon as key and label as value. If None loads a default map
        :param color_map: Color Map for each of room and icon type. If None loads random colors
        """

        self.wall_thickness = wall_thickness
        self.window_thickness = window_thickness
        self.door_thickness = door_thickness
        self.corner_thickness = corner_thickness
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.out_width = out_width
        self.out_height = out_height
        self.out_format = out_format

        if room_map is None:
            self.room_map = self._get_default_room_label_map()
        else:
            self.room_map = room_map

        if icon_map is None:
            self.icon_map = self._get_default_icon_label_map()
        else:
            self.icon_map = icon_map

        self.label_map = self._load_default_label_map()

        if color_map is None:
            self.color_map = self._load_default_color_map()
        else:
            self.color_map = color_map

    # Default room label map
    @staticmethod
    def _get_default_room_label_map():
        return {
            'living_room': 1, 'kitchen': 2, 'bedroom': 3,
            'bathroom': 4, 'restroom': 4, 'washing_room': 4,
            'office': 3, 'closet': 6, 'balcony': 7,
            'corridor': 8, 'dining_room': 9, 'laundry_room': 10,
            'PS': 10
        }

    # Default icon label map
    @staticmethod
    def _get_default_icon_label_map():
        return {
            'bathtub': 1, 'cooking_counter': 2,
            'toilet': 3, 'entrance': 4,
            'washing_basin': 5, 'special': 6,
            'stairs': 7
        }

    # Loads label map from room map and icon map
    def _load_default_label_map(self):

        label_map = {}
        for icon, label in self.icon_map.items():
            label_map[icon] = ('icons', label)
            continue
        for room, label in self.room_map.items():
            label_map[room] = ('rooms', label)
            continue
        label_map['door'] = 8

        return label_map

    # Loads random colors for rooms and icons
    def _load_default_color_map(self):

        num_colors = max(len(self.icon_map), len(self.room_map), 17)
        colors = np.random.randint(255, size=(num_colors, 3))

        return colors


class VectorToImageTransformer:

    def __init__(self, config):

        self.config = config
        self.inp_dir = config.inp_dir
        self.out_dir = config.out_dir
        self.out_format = config.out_format
        self.wall_thickness = config.wall_thickness
        self.out_width = config.out_width
        self.out_height = config.out_height
        self.colors = self.config.color_map

    def transform_vectors_to_images(self, shuffle_data=False, num_images=-1):

        create_directory_if_not_exist(self.out_dir)
        file_paths = self._get_file_paths(num_images, shuffle_data)

        if len(file_paths) == 0:
            print('No vector files found in the path. Please check the path')
            exit(1)

        for index, file_path in enumerate(tqdm(file_paths)):

            data = self._parse_vector_file(file_path)

            if not self._is_valid(data['rooms'], data['icons']):
                continue

            self._transform_points(data)
            data['wall'] = self._filter_walls(data['wall'], data['wall_types'])
            corners, success = self._lines_2_corners(data['wall'], gap=self.config.wall_thickness)

            shape_mask = self._get_bounding_mask(data['wall'])
            wall_mask = self._get_wall_mask(data['wall'])
            door_mask, window_mask, dc, wc = self._get_door_window_mask(data['window'], data['door'], shape_mask)
            room_mask, room_types = self._get_room_mask(data['wall'], data['rooms'], shape_mask)
            corner_mask = self._get_corner_mask(data['door'], corners)
            entrance_mask = self._get_icon_mask(data['icons'])

            masks = {
                'wall_mask': wall_mask,
                'door_mask': door_mask,
                'window_mask': window_mask,
                'entrance_mask': entrance_mask,
                'room_mask': room_mask,
                'shape_mask': shape_mask,
                'corner_mask': corner_mask,
                'room_types': room_types
            }

            meta_data = {
                'wall_count': len(data['wall']),
                'door_count': dc,
                'window_count': wc,
                'cooling': data['cooling'],
                'heating': data['heating']
            }

            if self.out_format == 'h5':
                self._store_as_h5(masks, meta_data, file_path)
            elif self.out_format == 'tfrecord':
                self._store_as_tfrecord(masks, meta_data, file_path)
            elif self.out_format == 'png':
                self._store_as_png(masks, file_path)

    @staticmethod
    def _get_base_directory(file_path):

        dir_1 = os.path.split(os.path.split(file_path)[0])
        dir_2 = os.path.split(dir_1[0])
        dir_3 = os.path.split(dir_2[0])

        return os.path.join(dir_3[1], dir_2[1], dir_1[1])

    def _store_as_tfrecord(self, masks, meta_data, file_path):
        """
        Stores the data into a tfrecord

        :param masks: Dictionary containing all the masks (wall, door, window, room, shape, corner, entrance, room_types
        :param meta_data: Dictionary containing wall count, door count, window count, cooling and heating
        :param file_path: Path to the input vector file which is used to extract the output path
        """

        base_dir = os.path.join(self.out_dir, self._get_base_directory(file_path))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        out_file_name = os.path.join(base_dir, os.path.splitext(os.path.basename(file_path))[0] + '.tfrecord')

        with tf.io.TFRecordWriter(out_file_name, 'GZIP') as writer:
            writer.write(serialize_floor_plan(*self._convert_to_tensors(masks, meta_data)))

    @staticmethod
    def _convert_to_tensors(masks, meta_data):

        for key, val in masks.items():
            masks[key] = tf.convert_to_tensor(val, dtype=tf.float32)
            if masks[key].shape.ndims == 3:
                masks[key] = tf.transpose(masks[key], perm=(1, 2, 0))

        return masks, meta_data

    def _store_as_png(self, masks, file_path):

        """
        Stores the segmentation images as png
        :param masks: Dictionary containing all the required masks
        :param file_path: Path to the input vector file to extract the output path
        """

        base_dir = os.path.join(self.out_dir, self._get_base_directory(file_path))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        out_file_prefix = os.path.join(base_dir, os.path.splitext(os.path.basename(file_path))[0])

        cv2.imwrite(out_file_prefix + '_wall.png', self._mask_to_segmentation_image(masks['wall_mask']))
        cv2.imwrite(out_file_prefix + '_window.png', self._mask_to_segmentation_image(masks['window_mask']))
        cv2.imwrite(out_file_prefix + '_entrance.png', self._mask_to_segmentation_image(masks['entrance_mask']))
        cv2.imwrite(out_file_prefix + '_door.png', self._mask_to_segmentation_image(masks['door_mask']))
        cv2.imwrite(out_file_prefix + '_room.png', self._mask_to_segmentation_image(masks['room_mask']))
        cv2.imwrite(out_file_prefix + '_corner.png', self._mask_to_segmentation_image(masks['corner_mask']))
        cv2.imwrite(out_file_prefix + '_shape.png', self._mask_to_segmentation_image(masks['shape_mask']))

    def _store_as_h5(self, masks, meta_data, file_path):

        """
        Stores the masks and segmentation images into h5py files
        :param masks: Dictionary containing all the required masks
        :param meta_data: Dictionary containing all real values like wall count etc.
        :param file_path: Path to the input vector file to extract the output path
        """
        base_dir = os.path.join(self.out_dir, self._get_base_directory(file_path))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        out_file_name = os.path.join(base_dir, os.path.splitext(os.path.basename(file_path))[0] + '.h5py')

        h5f = h5py.File(out_file_name, 'w')
        for key, value in masks.items():
            h5f.create_dataset(key, data=value)
            if 'mask' in key:
                h5f.create_dataset(key.replace('mask', 'seg'), data=value)

        for key, value in meta_data.items():
            h5f.create_dataset(key, data=value)

        h5f.close()

    def _get_file_paths(self, num_images, shuffle_data):

        if '.txt' in self.inp_dir:
            file_paths = [self.inp_dir]
        else:
            file_paths = glob.glob(self.inp_dir + '/*/*/*/*.txt')

        if shuffle_data:
            shuffle(file_paths)

        if len(file_paths) > num_images > 0:
            file_paths = file_paths[:num_images]
            return file_paths

        return file_paths

    def _parse_vector_file(self, file_path):

        out = {
            'wall': [], 'door': [], 'window': [], 'wall_types': [],
            'rooms': {}, 'icons': {},
            'cooling': 0, 'heating': 0,
            'max_x': -sys.maxsize, 'max_y': -sys.maxsize,
            'min_x': sys.maxsize, 'min_y': sys.maxsize,
        }

        with open(file_path) as vector:

            for line in vector:
                line = line.split('\t')
                label = line[4].strip()

                if label in ['wall', 'door', 'window'] or label in self.config.label_map.keys():
                    if label in ['wall', 'door', 'window']:
                        out[label].append((convert_to_point(line[0], line[1]), convert_to_point(line[2], line[3])))
                        if label == 'wall':
                            out['wall_types'].append(int(line[5].strip()) - 1)
                    elif label in self.config.label_map.keys():
                        group = self.config.label_map[label][0]
                        if group == 'rooms':
                            if label not in out['rooms']:
                                out['rooms'][label] = []

                            out['rooms'][label].append(
                                (convert_to_point(line[0], line[1]), convert_to_point(line[2], line[3])))

                        if group == 'icons':
                            if label not in out['icons']:
                                out['icons'][label] = []

                            out['icons'][label].append(
                                (convert_to_point(line[0], line[1]), convert_to_point(line[2], line[3])))
                    out['max_x'] = max(out['max_x'], int(round(float(line[0]))), int(round(float(line[2]))))
                    out['max_y'] = max(out['max_y'], int(round(float(line[1]))), int(round(float(line[3]))))
                    out['min_x'] = min(out['min_x'], int(round(float(line[0]))), int(round(float(line[2]))))
                    out['min_y'] = min(out['min_y'], int(round(float(line[1]))), int(round(float(line[3]))))
                elif label == 'cooling':
                    out['cooling'] = float(line[0])
                elif label == 'heating':
                    out['heating'] = float(line[0])

        return out

    @staticmethod
    def _is_valid(rooms, icons):

        if 'stairs' in icons or 'entrance' not in icons:
            return False

        return True

    def _filter_walls(self, walls, wall_types):

        invalid_indices = {}
        for wall_index_1, (wall_1, wall_type_1) in enumerate(zip(walls, wall_types)):
            for wall_index_2, (wall_2, wall_type_2) in enumerate(zip(walls, wall_types)):
                if wall_index_1 in invalid_indices or wall_index_2 in invalid_indices:
                    continue
                if check_if_walls_touch(wall_1, wall_2, self.config.wall_thickness) and wall_index_1 < wall_index_2:
                    walls[wall_index_1] = merge_lines(wall_1, wall_2)
                    invalid_indices[wall_index_2] = True

                if wall_type_1 == 0 and wall_type_2 == 1 and calc_line_direction(wall_1) == calc_line_direction(wall_2):
                    if min([p2p_distance(wall_1[c_1], wall_2[c_2]) for c_1, c_2 in
                            [(0, 0), (0, 1), (1, 0), (1, 1)]]) <= self.config.wall_thickness * 2:
                        walls[wall_index_1] = merge_lines(wall_1, wall_2)
                        invalid_indices[wall_index_2] = True

        return [wall for wall_index, wall in enumerate(walls) if wall_index not in invalid_indices]

    @staticmethod
    def _find_connections(line_1, line_2, gap):
        connection_1 = -1
        connection_2 = -1
        point_connected = False
        for c_1 in range(2):
            if point_connected:
                break
            for c_2 in range(2):
                if p2p_distance(line_1[c_1], line_2[c_2]) > gap:
                    continue

                connection_1 = c_1
                connection_2 = c_2
                connection_point = ((line_1[c_1][0] + line_2[c_2][0]) // 2, (line_1[c_1][1] + line_2[c_2][1]) // 2)
                point_connected = True
                break
            continue
        if point_connected:
            return [connection_1, connection_2], connection_point

        direction_1, fixed_value_1, min_1, max_1 = line_range(line_1)
        direction_2, fixed_value_2, min_2, max_2 = line_range(line_2)
        if direction_1 == direction_2:
            return [-1, -1], (0, 0)

        # print(fixedValue_1, min_1, max_1, fixedValue_2, min_2, max_2)
        if min(fixed_value_1, max_2) < max(fixed_value_1, min_2) - gap or min(fixed_value_2, max_1) < max(fixed_value_2,
                                                                                                          min_1) - gap:
            return [-1, -1], (0, 0)

        if abs(min_1 - fixed_value_2) <= gap:
            return [0, 2], (fixed_value_2, fixed_value_1)
        if abs(max_1 - fixed_value_2) <= gap:
            return [1, 2], (fixed_value_2, fixed_value_1)
        if abs(min_2 - fixed_value_1) <= gap:
            return [2, 0], (fixed_value_2, fixed_value_1)
        if abs(max_2 - fixed_value_1) <= gap:
            return [2, 1], (fixed_value_2, fixed_value_1)

        return [2, 2], (fixed_value_2, fixed_value_1)

    def _lines_2_corners(self, lines, gap):
        success = True
        line_connections = []
        for _ in range(len(lines)):
            line_connections.append({})
            continue

        connection_corner_map = {
            (1, 1): 4, (0, 1): 5, (0, 0): 6, (1, 0): 7,
            (2, 0): 8, (1, 2): 9, (2, 1): 10, (0, 2): 11, (2, 2): 12
        }
        corners = []
        for lineIndex_1, line_1 in enumerate(lines):
            for lineIndex_2, line_2 in enumerate(lines):

                if lineIndex_2 == lineIndex_1:
                    continue

                connections, connection_point = self._find_connections(line_1, line_2, gap=gap)
                if connections[0] == -1 and connections[1] == -1:
                    continue

                # Lines overlap check
                if calc_line_direction(line_1) == calc_line_direction(line_2) and \
                        is_manhattan(line_1) and is_manhattan(line_2):
                    continue

                if calc_line_direction(line_1) == 1:
                    continue

                indices = [lineIndex_1, lineIndex_2]
                for c in range(2):
                    # Duplicate corner check
                    if connections[c] in [0, 1] \
                            and connections[c] in line_connections[indices[c]] \
                            and is_manhattan(line_1) and is_manhattan(line_2):
                        continue

                    line_connections[indices[c]][connections[c]] = True
                    continue
                corners.append((connection_point, connection_corner_map[tuple(connections)]))

        for index, map in enumerate(line_connections):
            if 0 not in map:
                line_direction = calc_line_direction(lines[index])
                if line_direction == 0:
                    corners.append((lines[index][0], int(lines[index][0][0] < lines[index][1][0])))
                if line_direction == 1:
                    corners.append((lines[index][0], 1 + int(lines[index][0][1] < lines[index][1][1])))

            if 1 not in map:
                line_direction = calc_line_direction(lines[index])
                if line_direction == 0:
                    corners.append((lines[index][1], int(lines[index][1][0] < lines[index][0][0])))
                if line_direction == 1:
                    corners.append((lines[index][1], 1 + int(lines[index][1][1] < lines[index][0][1])))

        return corners, success

    @staticmethod
    def _augment_point(point, scale_factor, tx_before, ty_before, tx_after, ty_after):
        """
        Scales and translates the point
        :param point: Point to be transformed
        :param scale_factor: Scale factor for scaling the point
        :param tx: Translation along x
        :param ty: Translation along y
        :return: Transformed point
        """

        return int(scale_factor * (point[0] + tx_before) + tx_after), \
               int(scale_factor * (point[1] + ty_before) + ty_after)

    def _transform_points(self, data):
        """
        Scales the points to fit into out_width, out_height
        Translates the points to the center of the output image
        """

        min_x, min_y, max_x, max_y = data['min_x'], data['min_y'], data['max_x'], data['max_y']

        tx_b = -min_x
        ty_b = -min_y

        sf = min(self.out_width / (max_y - min_y), self.out_height / (max_x - min_x)) * 0.85

        tx_a = (self.out_height / 2 - (max_x - min_x) * sf / 2)
        ty_a = (self.out_width / 2 - (max_y - min_y) * sf / 2)

        data['wall'] = self._augment_points(data['wall'], sf, tx_b, ty_b, tx_a, ty_a)
        data['door'] = self._augment_points(data['door'], sf, tx_b, ty_b, tx_a, ty_a)
        data['window'] = self._augment_points(data['window'], sf, tx_b, ty_b, tx_a, ty_a)

        for label, items in data['icons'].items():
            data['icons'][label] = self._augment_points(items, sf, tx_b, ty_b, tx_a, ty_a)
        for label, items in data['rooms'].items():
            data['rooms'][label] = self._augment_points(items, sf, tx_b, ty_b, tx_a, ty_a)

        return data

    def _augment_points(self, lines: list, scale, tx_b, ty_b, tx_a, ty_a):

        return [[self._augment_point(window[c], scale, tx_b, ty_b, tx_a, ty_a) for c in range(2)] for window in lines]

    def _get_bounding_mask(self, walls):

        room_segmentation = np.zeros((self.out_height, self.out_width), dtype=np.uint8)

        for line in walls:
            cv2.line(room_segmentation, line[0], line[1], color=128, thickness=self.config.wall_thickness)

        # Makes the walls as the background
        rooms = measure.label(room_segmentation == 0, background=0)
        wall_label = rooms.min()
        for pixel in [(0, 0), (0, self.out_height - 1), (self.out_width - 1, 0),
                      (self.out_width - 1, self.out_height - 1)]:
            background_label = rooms[pixel[1]][pixel[0]]
            if background_label != wall_label:
                break
            continue

        bounding_mask = np.ones((1, self.out_height, self.out_width), dtype=np.uint8)
        bounding_mask[0][rooms == background_label] = 0

        return bounding_mask

    def _get_wall_mask(self, walls):
        """
        Generates mask for the walls from the coordinates of the walls

        :param walls: List of tuple of end poins of walls
        :return: Mask with 1 where walls are present and 0 otherwise
        """
        mask = np.zeros((1, self.out_height, self.out_width), dtype=np.uint8)
        for line in walls:
            cv2.line(mask[0], line[0], line[1], color=1, thickness=self.config.wall_thickness)

        return mask

    def _get_door_window_mask(self, windows, doors, shape_mask):
        """
        Generates mask for the windows and doors from the coordinates of the windows and doors
        And door that is on the boundary of the building is considered as a window

        :param windows: List of tuple of end points of windows
        :param doors: List of tuple of end points of doors
        :param shape_mask: Mask of the shape of the building
        :return: Doors mask and window mask
        """
        window_mask = np.zeros((1, self.out_height, self.out_width), dtype=np.uint8)
        door_mask = np.zeros((1, self.out_height, self.out_width), dtype=np.uint8)

        for line in windows:
            cv2.line(window_mask[0], line[0], line[1], color=1, thickness=self.config.wall_thickness)

        boundary = cv2.Laplacian(shape_mask[0], cv2.CV_64F)
        boundary = cv2.GaussianBlur(boundary, (11, 11), 0)
        boundary[boundary != 0] = 1

        door_count = 0
        window_count = 0

        for door in doors:
            if boundary[door[0][1]][door[0][0]] == 1 and boundary[door[1][1]][door[1][0]] == 1:
                cv2.line(window_mask[0], door[0], door[1], color=1, thickness=self.config.window_thickness)
                window_count += 1
            else:
                cv2.line(door_mask[0], door[0], door[1], color=1, thickness=self.config.door_thickness)
                door_count += 1

        return door_mask, window_mask, door_count, window_count

    def _get_door_mask(self, doors):
        """
        Generates mask for the doors from the coordinates of the doors

        :param doors: List of tuple of end points of doors
        :return: Mask with 1 where doors are present and 0 otherwise
        """
        mask = np.zeros((1, self.out_height, self.out_width), dtype=np.uint8)
        for line in doors:
            cv2.line(mask[0], line[0], line[1], color=1, thickness=self.config.wall_thickness)

        return mask

    def _get_window_mask(self, windows, doors, shape_mask, door_mask):
        """
        Generates mask for the windows from the coordinates of the windows

        :param windows: List of tuple of end points of windows
        :return: Mask with 1 where windows are present and 0 otherwise
        """
        mask = np.zeros((1, self.out_height, self.out_width), dtype=np.uint8)
        for line in windows:
            cv2.line(mask[0], line[0], line[1], color=1, thickness=self.config.wall_thickness)

        boundary = cv2.Laplacian(shape_mask[0], cv2.CV_64F)
        boundary = cv2.GaussianBlur(boundary, (11, 11), 0)
        boundary[boundary != 0] = 1

        for door in doors:
            if boundary[door[0][1]][door[0][0]] == 1 and boundary[door[1][1]][door[1][0]] == 1:
                cv2.line(mask[0], door[0], door[1], color=1, thickness=self.config.wall_thickness)

        return mask

    def _get_icon_mask(self, icons):

        mask = np.zeros((1, self.out_height, self.out_width), dtype=np.uint8)

        if 'entrance' not in icons:
            return mask

        for icon in icons['entrance']:
            cv2.rectangle(mask[0], icon[0], icon[1], color=1, thickness=self.config.wall_thickness)

        return mask

    def _get_corner_mask(self, doors, corners):
        all_corners = []
        for corner in corners:
            all_corners.append((corner[0][0], corner[0][1], corner[1]))

        opening_corner_map = [[3, 1], [0, 2]]

        for opening in doors:
            direction = calc_line_direction(opening)
            for cornerIndex, corner in enumerate(opening):
                all_corners.append(
                    (int(round(corner[0])),
                     int(round(corner[1])),
                     13 + opening_corner_map[direction][cornerIndex]))

        mask = np.zeros((17, self.out_height, self.out_width), dtype=np.uint8)

        for corner in all_corners:
            cv2.circle(mask[corner[2]], (corner[0], corner[1]),
                       radius=self.config.corner_thickness, color=1, thickness=-1)

        return mask

    def _get_room_mask(self, walls, rooms, shape_mask):
        """
        Gets the room masks for each type of room

        :param walls: List of lines(2 points) which represent walls in the floor plan
        :param rooms: A dict of label and list of rooms(2 coordinates) of that type
        :return: Image mask for all the types of rooms
        """
        # Draws lines where walls are
        wall_mask = shape_mask[0].copy()
        for line in walls:
            cv2.line(wall_mask, line[0], line[1], color=128, thickness=self.config.wall_thickness)

        # Makes the walls as the background
        room_segmentation = measure.label(wall_mask == 1, background=0)

        total_rooms = (max(self.config.room_map.values()) - min(self.config.room_map.values()) + 1)
        mask = np.zeros((total_rooms, self.out_height, self.out_width), dtype=np.uint8)
        room_types = np.zeros(total_rooms, dtype=np.uint8)

        for label, items in rooms.items():

            for point in items:
                room_index = room_segmentation[(point[0][1] + point[1][1]) // 2][(point[0][0] + point[1][0]) // 2]
                if room_index == 0:
                    continue
                mask_index = self.config.room_map[label] - min(self.config.room_map.values())
                mask[mask_index][room_segmentation == room_index] = 1
                room_types[mask_index] = 1

        return mask, room_types

    def _mask_to_segmentation_image(self, mask):
        """
        Converts a mask image with dimensions HxWxC into a segmentation image with C labels

        :param mask: A HXWXC mask image
        :return: A segmentation image with C labels
        """
        num_colors = mask.shape[0]
        if num_colors == 1:
            colors = [0]
        else:
            colors = self.colors

        # print(self.colors, num_colors)

        segmentation_image = 255 * np.ones((mask.shape[1], mask.shape[2], 3), np.uint8)

        for i in range(mask.shape[0]):
            segmentation_image[mask[i] == 1] = colors[i]

        return segmentation_image
