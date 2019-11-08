#!/usr/bin/env python
# coding: utf-8

import numpy as np

from IP import reconstructFloorplan

## This should be replaced by the output of the GAN
## with the shapes below as ///// heatmaps //////
### wall Corners --> 256x256x13
### door Corners --> 256x256x4
### room Types   --> 256x256x10


wallCornerHeatmaps = np.load('input/example_wallCornerHeatmaps.npy')
doorCornerHeatmaps = np.load('input/example_doorCornerHeatmaps.npy')
roomHeatmaps = np.load('input/example_roomHeatmaps.npy')

print('{} has shape={}'.format('wallCornerHeatmaps',wallCornerHeatmaps.shape))
print('{} has shape={}'.format('doorCornerHeatmaps',doorCornerHeatmaps.shape))
print('{} has shape={}'.format('roomHeatmaps',roomHeatmaps.shape))


## Running the IP
### output_prefix (type str )= where to save text file, can be set to None
### save_image (type bool )= save image or not
ip_result = reconstructFloorplan(wallCornerHeatmaps,
                         doorCornerHeatmaps,
                         roomHeatmaps,
                         output_prefix='output/example_',
                         save_image=True)


#print('*********** WALL ***********',ip_result['wall'])


#print('*********** DOOR ***********',ip_result['door'])
