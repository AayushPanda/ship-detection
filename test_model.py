import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from gen_mask import gen_mask

imageX = 768
imageY = 768

data = pd.DataFrame(pd.read_csv("data\\data_segmentations.csv"))
data_dir = "data\\images\\"
fileID = "000155de5.jpg"

model = keras.models.load_model('model.h5', compile=False)

# Test on an image
test_img = np.expand_dims(cv2.imread(data_dir + fileID, cv2.IMREAD_COLOR), axis=0)

prediction = model.predict(test_img)

plt.imshow(cv2.imread(data_dir + fileID, cv2.IMREAD_COLOR))
plt.show()

plt.imshow(np.squeeze(prediction))
plt.show()