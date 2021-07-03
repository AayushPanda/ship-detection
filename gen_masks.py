import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.DataFrame(pd.read_csv("data\\data_segmentations.csv", nrows=1000))


def gen_mask(img_id):
    img_location = "C:\\Users\\Aayush\\Documents\\GitHub\\ship-detection\\data\\images\\" + img_id

    en_pix = str(data[data.ImageId == img_id].EncodedPixels.item())

    rle = list(map(int, en_pix.split()))

    pixel, pixel_count = [], []
    [pixel.append(rle[i]) if i % 2 == 0 else pixel_count.append(rle[i]) for i in range(0, len(rle))]

    rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(0, len(pixel))]

    rle_mask_pixels = sum(rle_pixels, [])

    mask_img = np.zeros((768 * 768, 1), dtype=int)

    mask_img[rle_mask_pixels] = 255

    l, b = cv2.imread(img_location).shape[0], cv2.imread(img_location).shape[1]
    mask = np.reshape(mask_img, (b, l)).T

    return mask



