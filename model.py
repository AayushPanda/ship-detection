import glob
import os

import cv2
import numpy as np
import pandas as pd
import segmentation_models as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

nData = 2000

imageX = 768
imageY = 768

data = pd.DataFrame(pd.read_csv("data\\data_segmentations.csv"))

def gen_mask(img_id):
    img_location = "C:\\Users\\Aayush\\Documents\\GitHub\\ship-detection\\data\\images\\" + img_id

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


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

data_dir = "data\\images\\"

# train_images = np.array(data.ImageId.apply(lambda file: cv2.imread(data_dir + file, cv2.IMREAD_COLOR)))

# Capture training image info as a list
train_images = []
train_masks = []

imgnum = 0

for directory_path in glob.glob("data\images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        imgnum += 1

        file_id = img_path[12:]
        print("Now generating: " + file_id + ' (' + str(imgnum) + ')')

        # Creating train masks array
        train_masks.append((gen_mask(file_id)))

        # Creating train images array
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (SIZE_Y, SIZE_X))tt
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        # train_labels.append(label)

        if imgnum >= nData:
            break

print("Masks and images generated")

# Convert list to array for machine learning processing
train_images = np.array(train_images)
train_masks = np.array(train_masks)

print("Converted to np array")

# Capture mask/label info as a list
# train_masks = []

# train_masks = np.array(data["EncodedPixels"].apply(str).apply(gen_mask))

# for directory_path in glob.glob("membrane/augmented_train_256/aug_mask"):
#     for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
#         mask = cv2.imread(mask_path, 0)
#         #mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
#         #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
#         train_masks.append(mask)
#         #train_labels.append(label)
# #Convert list to array for machine learning processing
# train_masks = np.array(train_masks)

# Use customary x_train and y_train variables
X = train_images
Y = train_masks
# Y = np.expand_dims(Y, axis=3)  # May not be necessary.. leftover from previous code

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['mse'])

print(model.summary())

history = model.fit(x_train,
                    y_train,
                    batch_size=8,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

model.save('model.h5')

accuracy = model.evaluate(x_val, y_val)
# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# from tensorflow import keras

# model = keras.models.load_model('membrane.h5', compile=False)
# Test on a different image
# READ EXTERNAL IMAGE...
# test_img = cv2.imread('membrane/test/0.png', cv2.IMREAD_COLOR)
# test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
# test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# test_img = np.expand_dims(test_img, axis=0)

# prediction = model.predict(test_img)

# View and Save segmented image
# prediction_image = prediction.reshape(mask.shape)
# plt.imshow(prediction_image, cmap='gray')
# plt.imsave('membrane/test0_segmented.jpg', prediction_image, cmap='gray')
#

# plt.imshow(gen_mask('000155de5.jpg'))
# plt.show()
