import cv2 as cv2
import numpy as np
from PIL import Image
import os
import SimpleITK as sitk

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from skimage.transform import resize
import glob
import openslide
import matplotlib.pyplot as plt
import xmltodict
import pandas as pd

#Rachel's code
import xml.etree.ElementTree as ET # to replace the xmltodict method
import tifffile as tiff


#From Rachel's Code
def extract_raw_annotation(annotation_file):
    with open(annotation_file, "r") as f:
        lines = f.readlines()

    num_annotations = sum(1 for line in lines if "ndpviewstate id=" in line)
    xs_list = [list() for _ in range(num_annotations)]
    ys_list = [list() for _ in range(num_annotations)]
    titles_list = [list() for _ in range(num_annotations)]
    colors_list = [list() for _ in range(num_annotations)]

    annotation_counter = 0

    for line_num, line in enumerate(lines):
        if line_num <= 1:
            pass
        elif "ndpviewstate id=" in line:
            xs_list[annotation_counter] = list()
            ys_list[annotation_counter] = list()
            annotation_counter += 1
        else:
            if "<title>" in line:
                titles_list[annotation_counter - 1] = line.strip()[7:-8]
            if "<x>" in line:
                xs_list[annotation_counter - 1].append(int(line.strip()[3:-4]))
            if "<y>" in line:
                ys_list[annotation_counter - 1].append(int(line.strip()[3:-4]))
            if "color" in line:
                colors_list[annotation_counter - 1] = line.strip()[-9:-2]

    for annotation_xs, annotation_ys in zip(xs_list, ys_list):
        annotation_xs.pop(0)
        annotation_ys.pop(0)

    return num_annotations, xs_list, ys_list, titles_list, colors_list

# From Rachel's Code
def extract_image_data(image_file, pyramid_tier):
    # Openslide Image Info
    tif_openslide = openslide.OpenSlide(image_file)
    x_offset = float(tif_openslide.properties["hamamatsu.XOffsetFromSlideCentre"])
    y_offset = float(tif_openslide.properties["hamamatsu.YOffsetFromSlideCentre"])
    x_mpp = float(tif_openslide.properties["openslide.mpp-x"])
    y_mpp = float(tif_openslide.properties["openslide.mpp-y"])
    level_dims = tif_openslide.level_dimensions
    source_lens = float(tif_openslide.properties["hamamatsu.SourceLens"])

    # retrieve array with tifffile - this is done to save memory
    tif = tiff.TiffFile(image_file)
    tier_array = np.asarray(tif._series_svs()[pyramid_tier].asarray())
    image_shape = tier_array.shape

    return (
        tier_array,
        image_shape,
        pyramid_tier,
        x_offset,
        y_offset,
        x_mpp,
        y_mpp,
        level_dims,
        source_lens,
    )

# From Rachel's Code
def calc_level_resolution(image, pyramid_tier):
    (tier_array, image_shape, pyramid_tier, x_offset, y_offset, x_mpp, y_mpp, level_dims, source_lens) = extract_image_data(image, pyramid_tier)

    num_levels = len(level_dims)
    x_res = np.zeros(num_levels)  # microns per pixel
    y_res = np.zeros(num_levels)  # microns per pixel
    zoom = np.zeros(num_levels)
    zoom[
        0
    ] = source_lens  # highest res pyramid level has the zoom of the source lens - 20x

    im_width_microns = x_mpp * level_dims[0][0]
    im_height_microns = y_mpp * level_dims[0][1]

    for i in range(num_levels):
        x_res[i] = im_width_microns / level_dims[i][0]
        y_res[i] = im_height_microns / level_dims[i][1]

    return x_res[pyramid_tier], y_res[pyramid_tier]

# From Rachel's Code
# convert annotation coords from nm to pixels
def convert_annotation_coords(image_file, annotation_file, pyramid_tier):
    (
        tier_array,
        image_shape,
        pyramid_tier,
        x_offset,
        y_offset,
        x_mpp,
        y_mpp,
        level_dims,
        source_lnes,
    ) = extract_image_data(image_file, pyramid_tier)
    num_annotations, xs_list, ys_list, _, _ = extract_raw_annotation(annotation_file)
    x_res, y_res = calc_level_resolution(image_file, pyramid_tier)
    new_x_coord = [list() for num in range(num_annotations)]
    new_y_coord = [list() for num in range(num_annotations)]


    for i, (annotation_xs, annotation_ys) in enumerate(zip(xs_list, ys_list)):
        new_x_coord[i] = list()
        new_y_coord[i] = list()
        for j, (x_coord, y_coord) in enumerate(zip(annotation_xs, annotation_ys)):
            new_x_coord[i].append(
                ((x_coord - x_offset) / (x_res * 1000)) + (image_shape[1] / 2)
            )
            new_y_coord[i].append(
                ((y_coord - y_offset) / (y_res * 1000)) + (image_shape[0] / 2)
            )
    return new_x_coord, new_y_coord #are these in units of pixels or in nm


#img used to crop the ROI from it, contour to get the vertices coordinates, start/end_x/y to calc patch size and give parameter for croping
def get_annotation_contour(img, contour, down_rate, shift, lv, start_x, start_y, end_x, end_y, resize_flag):
    vertices = contour['Vertices']['Vertex']

    print('vertices values are', vertices)
    
    cnt = np.zeros((4,1,2))

    now_id = contour['@Id']

    #for vertices:1st index(0 for x, 1 for y), 2nd index(which annoation), 3rd index(which point)
    cnt[0, 0, 0] = vertices[0][now_id-1][0]
    cnt[0, 0, 1] = vertices[1][now_id-1][0]
    cnt[1, 0, 0] = vertices[0][now_id-1][1]
    cnt[1, 0, 1] = vertices[1][now_id-1][1]
    cnt[2, 0, 0] = vertices[0][now_id-1][2]
    cnt[2, 0, 1] = vertices[1][now_id-1][2]
    cnt[3, 0, 0] = vertices[0][now_id-1][3]
    cnt[3, 0, 1] = vertices[1][now_id-1][3]

    print('shift value:', shift)

    cnt[0, 0, 0] = cnt[0, 0, 0] - shift
    cnt[1, 0, 0] = cnt[1, 0, 0] - shift
    cnt[2, 0, 0] = cnt[2, 0, 0] - shift
    cnt[3, 0, 0] = cnt[3, 0, 0] - shift

    cnt = cnt.astype(int)

    print(cnt)

    print('down rate applied:', down_rate)

    patch_size_x = int(abs(cnt[2, 0, 0] - cnt[0, 0, 0]) / down_rate)
    patch_size_y = int(abs(cnt[2, 0, 1] - cnt[0, 0, 1]) / down_rate)


    #where is the origin (0, 0)? learn how to use vscode debugger
    patch_start_x = np.min(cnt[:,:,0]) + start_x
    patch_start_y = np.min(cnt[:,:,1]) + start_y
    print('patch_start_x:',patch_start_x, 'patch_start_y:',patch_start_y, 'patch_size_x:',patch_size_x, 'patch_size_y:', patch_size_y)
    print('level number:',lv)

    # where the memory issue occured
    patch = np.array(img.read_region((patch_start_x, patch_start_y), lv, (patch_size_x, patch_size_y)).convert('RGB'))

    

    if resize_flag:
        patch_resize = resize(patch, (int(patch.shape[0]/ 2), int(patch.shape[1] / 2)))
        cnt = cnt / 2
    else:
        patch_resize = patch

    return patch_resize, cnt, now_id

def scn_to_png(ndpi_file, annotation_ndpa_file, output_folder, single_annotation):
    simg = openslide.open_slide(ndpi_file)
    print('this is the size of the ndpi file', simg.dimensions)
    name = os.path.basename(ndpi_file).replace('.ndpi', '')

    #where to add the convert_annotation_contour 
    pixel_coord = convert_annotation_coords(ndpi_file, annotation_ndpa_file, 0) #pyrimid level 0 to 10(specific to openslide)
    print('covert corrd ouput:', pixel_coord)


    # read annotation region
    tree = ET.parse(annotation_ndpa_file)
    root = tree.getroot()
    ndpviewstates = root.findall('.//ndpviewstate')
    
    annotations =[]
    for ndpviewstate in ndpviewstates:
        annotation_id = int(ndpviewstate.get('id'))
        annotations.append(annotation_id)
    
    #ET.dump(tree.getroot())
    print(annotations)

    #with open(annotation_dnpa_file) as fd:
    #    annotation_doc = xmltodict.parse(fd.read())
    #annotation_layers = annotation_doc['Annotations']['Annotation']
    #try:
    #    annotation_contours = annotation_layers['Regions']['Region']
    #except:
    #    if len(annotation_layers) == 2:
    #        annotation_BBlayer = annotation_layers[0]
    #        annotation_regions = annotation_BBlayer['Regions']['Region']
    #        annotation_Masklayer = annotation_layers[1]
    #    else:
    #        annotation_Masklayer = annotation_layers[0]
    #    annotation_contours = annotation_Masklayer['Regions']['Region']


    # start_x, start_y = get_nonblack_starting_point(simg)
    end_x, end_y = 0, 0 #get_nonblack_ending_point(simg)
    #
    # print(start_x, start_y)
    # print(end_x,end_y)
    start_x, start_y = 0, 0
    if single_annotation:
        for annotation in annotations:
            print('annotation in loop', annotation)
            contour = {
                'Vertices': {
                'Vertex': pixel_coord
                },
                '@Id': annotation # Extracting id from the annotation element
            }


            patch_10X, cnt_10X, id = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 0)
            patch_40X, cnt_40X, _ = get_annotation_contour(simg, contour, simg.level_downsamples[0], 0, 0, start_x, start_y, end_x, end_y, 0)
            patch_5X, cnt_5X, _ = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 1)

            X40_output_folder = os.path.join(output_folder, '40X')
            X5_output_folder = os.path.join(output_folder, '5X')
            X10_output_folder = os.path.join(output_folder, '10X')

            if not os.path.exists(X40_output_folder):
                os.makedirs(X40_output_folder)

            if not os.path.exists(X5_output_folder):
                os.makedirs(X5_output_folder)

            if not os.path.exists(X10_output_folder):
                os.makedirs(X10_output_folder)

            now_name = '%s_%s.png' % (name, id)
            plt.imsave(os.path.join(X40_output_folder, now_name), patch_40X)
            plt.imsave(os.path.join(X5_output_folder, now_name), patch_5X)
            plt.imsave(os.path.join(X10_output_folder, now_name), patch_10X)
    else:
         for annotation in annotations:
            print('To keep track of which annotation is being done:', annotation)
            contour = {
                 'Vertices': {
                'Vertex': pixel_coord
                },
                '@Id': annotation
            }


            patch_10X, cnt_10X, id = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 0)
            patch_40X, cnt_40X, _ = get_annotation_contour(simg, contour, simg.level_downsamples[0], 0, 0, start_x, start_y, end_x, end_y, 0)
            patch_5X, cnt_5X, _ = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 1)

            X40_output_folder = os.path.join(output_folder, '40X')
            X5_output_folder = os.path.join(output_folder, '5X')
            X10_output_folder = os.path.join(output_folder, '10X')

            if not os.path.exists(X40_output_folder):
                os.makedirs(X40_output_folder)

            if not os.path.exists(X5_output_folder):
                os.makedirs(X5_output_folder)

            if not os.path.exists(X10_output_folder):
                os.makedirs(X10_output_folder)

            now_name = '%s_%s.png' % (name, id)
            plt.imsave(os.path.join(X40_output_folder, now_name), patch_40X)
            plt.imsave(os.path.join(X5_output_folder, now_name), patch_5X)
            plt.imsave(os.path.join(X10_output_folder, now_name), patch_10X)


if __name__ == '__main__':
    dirpath = '/home/ericxlinux/kidney_segmentation/Omni-Seg/Omni_seg_pipeline_gpu/svs_input'
    filename = 'BR22-2073-A-1-9-TRI - 2022-08-08 15.03.42.ndpi'

    # annotation file
    now_annotation_xml = 'BR22-2073-A-1-9-TRI - 2022-08-08 15.03.42.ndpi.ndpa'

    # single_annotation indicates that whether the .xml file only contain single region of annotation.
    scn_to_png(dirpath + '/' + filename, dirpath + '/' + now_annotation_xml, dirpath, single_annotation=False)