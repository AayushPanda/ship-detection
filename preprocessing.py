import glob
import os

import cv2
import numpy as np
import pandas as pd
import segmentation_models as sm
from sklearn.model_selection import train_test_split

# Defining constants
data = pd.DataFrame(pd.read_csv("data\\data_segmentations.csv"))
data_dir = "data\\images\\"

nData = 4000

imageX = 768
imageY = 768

test_size = 0.2
random_seed = 42

BACKBONE = 'resnet34'


# Function to generate mask from encoded pixels
def gen_mask(img_id):
    img_location = data_dir + img_id

    if type(img_id) == str or str(img_id) == "":
        return np.zeros((768, 768, 1), dtype=np.uint8)

    en_pix = str(data[data.ImageId == img_id].EncodedPixels.item())

    rle = list(map(int, en_pix.split()))

    pixel, pixel_count = [], []
    [pixel.append(rle[i]) if i % 2 == 0 else pixel_count.append(rle[i]) for i in range(0, len(rle))]

    rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(0, len(pixel))]

    rle_mask_pixels = sum(rle_pixels, [])

    mask_img = np.zeros((imageX * imageY), dtype=int)
    mask_img[rle_mask_pixels] = 255

    l, b = cv2.imread(img_location, 0).shape[0], cv2.imread(img_location).shape[1]
    mask = np.reshape(mask_img, (b, l, 1, 1, 1)).T

    return mask


# Capture training image info as a list
train_images = []
train_masks = []

img_num = 0

for directory_path in glob.glob(data_dir):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img_num += 1

        file_id = img_path[12:]
        print("Now generating: " + file_id + ' (' + str(img_num) + ')')

        # Creating train masks array
        train_masks.append((gen_mask(file_id)))

        # Creating train images array
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        train_images.append(img)

        if img_num >= nData:
            break

print("Masks and images generated")

# Convert list to numpy array to allow for usage by machine learning model
train_images = np.array(train_images)
train_masks = np.array(train_masks)

print("Converted to np array")

# Train-test split
X = train_images
Y = train_masks

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# preprocess input
preprocess_input = sm.get_preprocessing(BACKBONE)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
