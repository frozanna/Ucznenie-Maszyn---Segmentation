import numpy as np
import os
import glob                
import cv2
import math

# set_type = "test"
# folder = "./Dataset/Drill_" + set_type + "/"
set_type = "kat_90"
folder = "./Dataset/Drill_test/"
category = "Cropped_384/" + set_type + "/"
color = "color/"
color_back = "color_back/"
mask = "mask/"
dest = "final/"

def load_images_from_folder(f, c):
    images = []
    for filename in os.listdir(f + c):
        img = f + c + filename
        if img is not None:
            images.append(filename)
    return images


def resize(path, size_x, size_y):
    color_back_paths = load_images_from_folder(path, "color_back/")
    color_paths = load_images_from_folder(path, "color/")
    mask_paths = load_images_from_folder(path, "mask/")
    keypoints_path = path + set_type + "_keypoints"
    crop_path = path + set_type + "_crop"

    final_color_path = folder + "final/" + set_type + "/color/"
    if not os.path.exists(final_color_path):
        os.makedirs(final_color_path)
    final_color_back_path = folder + "final/" + set_type + "/color_back/"
    if not os.path.exists(final_color_back_path):
        os.makedirs(final_color_back_path)
    final_mask_path = folder + "final/" + set_type + "/mask/"
    if not os.path.exists(final_mask_path):
        os.makedirs(final_mask_path)
    final_keypoints_path = folder + "final/" + set_type + "/"
    final_crop_path = folder + "final/" + set_type + "/"
    
    file = open(keypoints_path, "r")
    file_content = file.read()
    keypoints = [line.split() for line in file_content.split('\n')]
    new_keypoints = []

    for i in range(len(mask_paths)):
        color = cv2.imread(path + "color/" + color_paths[i], 1)
        color_back = cv2.imread(path + "color_back/" + color_back_paths[i], 1)
        mask = cv2.imread(path + "mask/" + mask_paths[i])

        color_resized = cv2.resize(color, dsize=(size_x, size_y), interpolation=cv2.INTER_CUBIC)
        color_back_resized = cv2.resize(color_back, dsize=(size_x, size_y), interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(mask, dsize=(size_x, size_y), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(final_color_path, color_paths[i]), color_resized)
        cv2.imwrite(os.path.join(final_color_back_path, color_back_paths[i]), color_back_resized)
        cv2.imwrite(os.path.join(final_mask_path, mask_paths[i]), mask_resized)

        w = len(color)
        ratio = w / size_x
        col = []
        for k in range(len(keypoints[i])):
            col.append(min(int(math.floor(int(keypoints[i][k])/ratio)), size_x))
        new_keypoints.append(col)

        keypoint_file = open(final_keypoints_path + set_type + "_keypoints_final", "w")
        np.savetxt(keypoint_file, new_keypoints, fmt='%s', delimiter='\t', newline='\n')

# resize(folder + category, 128, 128)
print('done')        
print('\n')