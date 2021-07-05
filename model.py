import pandas as pd
import segmentation_models as sm
from matplotlib import pyplot as plt

from preprocessing import x_train, x_test, y_train, y_test

# Defining constants
batch_size = 1
train_epochs = 1
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
    model.save('model1.h5')

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

# 372s 465ms/sample - loss: 5.6272e-05 - mse: 3.4308e-09