import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Defining constants
imageX = 768
imageY = 768

data = pd.DataFrame(pd.read_csv("data\\data_segmentations.csv"))
data_dir = "data\\images\\"


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
