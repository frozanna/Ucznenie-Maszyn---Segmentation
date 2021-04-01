import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from augmentators import randomHueSaturationValue, randomHorizontalFlip, randomShiftScaleRotate
from u_net import get_unet_128
import glob

orig_width = 256
orig_height = 256

threshold = 0.5


epochs = 10
batch_size = 1
input_size, model = get_unet_128()
model.load_weights(filepath='myweights/best_weights2.hdf5')

print(input_size)


test_filenames = glob.glob(
    "input/Drill_test/Cropped_256/kat_30/color_back/img_*.png")
test_filenames = [filename.replace(
    '\\', '/').replace('.png', '') for filename in test_filenames]
test_filenames = [filename.split('/')[-1][4:] for filename in test_filenames]


print('Predicting on {} samples with batch_size = {}...'.format(
    len(test_filenames), batch_size))
for start in tqdm(range(0, len(test_filenames), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(test_filenames))
    ids_test_batch = test_filenames[start:end]
    for id in ids_test_batch:
        img = cv2.imread(
            'input/Drill_test/Cropped_256/kat_30/color/img_{}.png'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds, preds_keypoints = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    preds_keypoints = np.squeeze(preds_keypoints, axis=0)
    for j in range(9):
        prob = np.array(cv2.resize(preds_keypoints[:, :, j], (orig_width, orig_height))).astype(np.float32) * 255
        cv2.imwrite(
            'input/Drill_test/Cropped_256/kat_30/keypoints/keypoints_{}_{}.png'.format(id, j), prob)

    for index, pred in enumerate(preds):
        prob = np.array(cv2.resize(pred, (orig_width, orig_height))
                        > threshold).astype(np.float32) * 255
        current_filename = ids_test_batch[index]
        cv2.imwrite(
            'input/Drill_test/Cropped_256/kat_30/segmantation/mask_{}.png'.format(id), prob)


print("Done!")
