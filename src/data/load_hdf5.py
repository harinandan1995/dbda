import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

filename = './datasets/128/hdf5/00/3e/c5cd44c6da484479553e7da20696/0001.h5py'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())

    wall_mask = np.array(f.get('wall_mask'))
    door_mask = np.array(f.get('door_mask'))
    window_mask = np.array(f.get('window_mask'))
    room_mask = np.array(f.get('room_mask'))
    corner_mask = np.array(f.get('corner_mask'))
    shape_mask = np.array(f.get('shape_mask'))
    entrance_mask = np.array(f.get('entrance_mask'))


    room_seg = np.array(f.get('room_seg'))
    plt.imshow(wall_seg)
    plt.savefig(os.path.join('./out', 'dummy.png'))
    plt.show()
