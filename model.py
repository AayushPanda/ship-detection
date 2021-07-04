import pandas as pd
import segmentation_models as sm
from matplotlib import pyplot as plt

from preprocessing import x_train, x_test, y_train, y_test

# Defining constants
batch_size = 8
train_epochs = 5
save_model = True

data = pd.DataFrame(pd.read_csv("data\\data_segmentations.csv"))
data_dir = "data\\images\\"

BACKBONE = 'resnet34'

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['mse'])

print(model.summary())

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=train_epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

if save_model:
    model.save('model.h5')

print("Done training in: ")

# Plot the metrics during training
accuracy = model.evaluate(x_test, y_test)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Plot the plots
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test on a random datapoint

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


# 372s 465ms/sample - loss: 5.6272e-05 - mse: 3.4308e-09