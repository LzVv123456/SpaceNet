#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:46:53 2016
@author: avanetten
"""

'''
Transform SpaceNet geojson buidling labels data into raster masks.
Download data via:
    aws s3api get-object --bucket spacenet-dataset \
    --key AOI_1_Rio/processedData/processedBuildingLabels.tar.gz \
    --request-payer requester processedBuildingLabels.tar.gz
Download spacenet utilities from:
   https://github.com/SpaceNetChallenge/utilities/tree/master/python/spaceNet 
Spacenet utils are not used here, but are used in geojson_to_pixel_arr.py 
      and screate_distance_map.py
For further details, see:
    https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53
'''

import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import sys
import os
import cv2
from PIL import Image
####################
# EDIT THESE PATHS
# download spacenet utilities from:
#   https://github.com/SpaceNetChallenge/utilities/tree/master/python/spaceNet
#  spacenet utils are not used here, but are used in geojson_to_pixel_arr.py
#      and screate_distance_map.py
# Set data dir
spacenet_data_dir = '/Users/cindy/utilities/python/'
# This is the directory where this script is located
# spacenet_explore_dir = os.path.dirname(os.path.realpath(__file__))
spacenet_explore_dir = '/Users/cindy/utilities/python/spacenet/'
# exclore N images in 3band data
N_ims = 15
####################

# import packages
sys.path.extend([spacenet_explore_dir])
import geojson_to_pixel_arr, create_dist_map, create_building_mask, \
    plot_truth_coords, plot_building_mask, plot_dist_transform, \
    plot_all_transforms


#
#
#####################
#####################
## EDIT THESE VALUES
#
## download spacenet utilities from:
##    https://github.com/SpaceNetChallenge/utilities/tree/master/python/spaceNet
##  spacenet utils are not used here, but are used in geojson_to_pixel_arr.py
##      and screate_distance_map.py
#
# path_to_spacenet_utils = '/Users/avanetten/Documents/cosmiq/git/spacenet-utilities-master/python/'
##sys.path.extend([path_to_spacenet_utils])
##from spaceNetUtilities import geoTools as gT
#
##spacenet_explore_dir = '/Users/avanetten/Documents/cosmiq/git/ave/spacenet_explore/'
# spacenet_explore_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.extend([spacenet_explore_dir])
# import geojson_to_pixel_arr, create_dist_map, create_building_mask, \
#        plot_truth_coords, plot_building_mask, plot_dist_transform, \
#        plot_all_transforms
#
## set path to spacenet data
## acquire from: https://aws.amazon.com/public-datasets/spacenet/
# spacenet_data_dir = '/Users/avanetten/Documents/cosmiq/spacenet/data/spacenetFromAWS_08122016/processedData/spacenet_data/'
##spacenet_data_dir = '/Users/avanetten/Documents/cosmiq/blupr_net/' + 'spacenet_data/'
#
## get N images in 3band data
# N_ims = 15
#
####################
####################

def main():
    imDir = '/Users/cindy/utilities/python/footprint/RGB-PanSharpen'
    # imDir = os.path.join(spacenet_data_dir, 'AOI_3_Paris_Train/RGB-PanSharpen')
    # vecDir = os.path.join(spacenet_data_dir, 'AOI_3_Paris_Train/geoJson')
    vecDir = '/Users/cindy/utilities/python/footprint/geojson'
    imDir_out = os.path.join(spacenet_explore_dir, '3band')

    ground_truth_patches = []
    pos_val, pos_val_vis = 1, 255

    ########################
    # Create directories

    # coordsDir = spacenet_explore_dir + 'pixel_coords_mask/'
    coords_demo_dir = os.path.join(spacenet_explore_dir, 'pixel_coords_demo')
    img_dir = os.path.join(spacenet_explore_dir, 'img')
    maskDir = os.path.join(spacenet_explore_dir, 'building_mask')
    maskDir2 = os.path.join(spacenet_explore_dir, 'mask')
    maskDir_vis = os.path.join(spacenet_explore_dir, 'building_mask_vis')
    mask_demo_dir = os.path.join(spacenet_explore_dir, 'mask_demo')

    distDir = os.path.join(spacenet_explore_dir, 'distance_trans')
    dist_demo_dir = os.path.join(spacenet_explore_dir, 'distance_trans_demo')

    all_demo_dir = os.path.join(spacenet_explore_dir, 'all_demo')

    # make dirs
    for p in [imDir_out, coords_demo_dir, maskDir, maskDir_vis, mask_demo_dir,
              distDir, dist_demo_dir, all_demo_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

    # get input images and copy to working directory
    # print(imDir)
    rasterList = glob.glob(os.path.join(imDir, '*.tif'))
    # print(rasterList)
    for im_tmp in rasterList:
        shutil.copy(im_tmp, imDir_out)

    # Create masks and demo images
    pixel_coords_list = []
    for i, rasterSrc in enumerate(rasterList):
        print(i, "Evaluating", rasterSrc)

        # input_image = plt.imread(rasterSrc)  #
        input_image = cv2.imread(rasterSrc, 1)
        # plt.imshow(input_image)
        # plt.show()
        b, g, r = cv2.split(input_image)  # get b,g,r
        input_image = cv2.merge([r, g, b])  # switch it to rgb


        # get name root
        name_root0 = rasterSrc.split('/')[-1]
        [name_root0, ext] = os.path.splitext(name_root0)
        # remove 3band or 8band prefix
        # print(name_root0)
        name_root = name_root0[14:]
        # print(name_root)
        vectorSrc = os.path.join(os.path.join(vecDir,'buildings'), 'buildings' +name_root+ '.geojson')
        maskSrc = os.path.join(maskDir, name_root0 + '.tif')
        # plot_name = os.path.join(img_dir, name_root + '.png')
        # print(plot_name)
        # cv2.imwrite(plot_name, rasterSrc)

        ####################################################
        # pixel coords and ground truth patches
        pixel_coords, latlon_coords = \
            geojson_to_pixel_arr.geojson_to_pixel_arr(rasterSrc, vectorSrc,
                                                      pixel_ints=True,
                                                      verbose=False)
        pixel_coords_list.append(pixel_coords)

        plot_name = os.path.join(coords_demo_dir, name_root + '.png')
        patch_collection, patch_coll_nofill = \
            plot_truth_coords.plot_truth_coords(input_image, pixel_coords,
                                                figsize=(8, 8), plot_name=plot_name,
                                                add_title=False)
        ground_truth_patches.append(patch_collection)
        plt.close('all')
        ####################################################

        ####################################################
        # building mask
        outfile = os.path.join(maskDir, name_root0 + '.tif')
        outfile_vis = os.path.join(maskDir_vis, name_root0 + '.tif')

        # create mask from 0-1 and mask from 0-255 (for visual inspection)
        create_building_mask.create_building_mask(rasterSrc, vectorSrc,
                                                  npDistFileName=outfile,
                                                  burn_values=pos_val)
        create_building_mask.create_building_mask(rasterSrc, vectorSrc,
                                                  npDistFileName=outfile_vis,
                                                  burn_values=pos_val_vis)

        plot_name = os.path.join(mask_demo_dir, name_root + '.png')
        mask_image = plt.imread(outfile)  # cv2.imread(outfile, 0)
        plot_building_mask.plot_building_mask(input_image, pixel_coords,
                                              mask_image,
                                              figsize=(8, 8), plot_name=plot_name,
                                              add_title=False)
        plot_name2 = os.path.join(maskDir2, name_root + '.png')
        cv2.imwrite(plot_name2,mask_image)
        plt.close('all')
        ####################################################

        ####################################################
        # signed distance transform
        # remove 3band or 8band prefix
        outfile = os.path.join(distDir, name_root0 + '.npy')  # '.tif'
        create_dist_map.create_dist_map(rasterSrc, vectorSrc,
                                        npDistFileName=outfile,
                                        noDataValue=0, burn_values=pos_val,
                                        dist_mult=1, vmax_dist=1)
        # plot
        # plot_name = os.path.join(dist_demo_dir + name_root, '_no_colorbar.png')
        plot_name = os.path.join(dist_demo_dir, name_root0 + '.png')
        mask_image = plt.imread(maskSrc)  # cv2.imread(maskSrc, 0)
        dist_image = np.load(outfile)
        plot_dist_transform.plot_dist_transform(input_image, pixel_coords,mask_image,
                                                dist_image, figsize=(8, 8),
                                                plot_name=plot_name,
                                                add_title=False,
                                                colorbar=True)  # False)
        plt.close('all')
        ####################################################

        ####################################################
        # plot all transforms
        plot_name = os.path.join(all_demo_dir, name_root + '.png')  # + '_titles.png'
        # print(plot_name)
        mask_image = plt.imread(maskSrc)  # cv2.imread(maskSrc, 0)
        dist_image = np.load(outfile)
        plot_all_transforms.plot_all_transforms(input_image, pixel_coords, mask_image, dist_image,
                                                figsize=(8, 8), plot_name=plot_name, add_global_title=False,
                                                colorbar=False,
                                                add_titles=False,  # True,
                                                poly_face_color='orange', poly_edge_color='red',
                                                poly_nofill_color='blue', cmap='bwr')
        plt.close('all')
        ####################################################


###############################################################################
if __name__ == '__main__':
    main()