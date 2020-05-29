import os
import pickle

import cv2
import keras
import matplotlib.pyplot as plt
# from scipy.misc import imread
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

imageDimesions = (32, 32, 3)
labelFile = 'labels.csv'  # file with all names of classes
batch_size_val = 50  # how many to process together
steps_per_epoch_val = 2000
epochs_val = 10

### LOADING DATASET
data_dir = os.path.abspath('GTSRB-Training_fixed/GTSRB/Training')
os.path.exists(data_dir)


### Function to resize the images using open cv
def resize_cv(im):
    return cv2.resize(im, (32, 32), interpolation=cv2.INTER_LINEAR)


### Loading datset
count = 0
images = []
classNo = []
output = []
for dir in os.listdir(data_dir):
    if dir == '.DS_Store':
        continue

    inner_dir = os.path.join(data_dir, dir)
    csv_file = pd.read_csv(os.path.join(inner_dir, "GT-" + dir + '.csv'), sep=';')
    for row in csv_file.iterrows():
        img_path = os.path.join(inner_dir, row[1].Filename)
        img = cv2.imread(img_path)
        img = img[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
        img = resize_cv(img)
        img = img / float(255)
        images.append(img)
        classNo.append(count)
        output.append(row[1].ClassId)
    print(count, end=" ")
    count += 1
### Plotting the dataset
fig = sns.distplot(output, kde=False, bins=43, hist=True, hist_kws=dict(edgecolor="black", linewidth=2))
fig.set(title="Traffic signs frequency graph",
        xlabel="ClassId",
        ylabel="Frequency")
# plt.show()

images = np.stack(images)
classNo = np.array(classNo)
train_y = keras.utils.np_utils.to_categorical(output)
y = train_y
############################### Split Data
split_size = int(images.shape[0] * 0.6)
train_x, val_x = images[:split_size], images[split_size:]
train1_y, val_y = y[:split_size], y[split_size:]

split_size = int(val_x.shape[0] * 0.5)
val_x, test_x = val_x[:split_size], val_x[split_size:]
val_y, test_y = val_y[:split_size], val_y[split_size:]


############################## READ CSV FILE
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

### Building the model
hidden_num_units = 2048
hidden_num_units1 = 1024
hidden_num_units2 = 128
output_num_units = 43

epochs = 10
batch_size = 16
pool_size = (2, 2)
input_shape = Input(shape=(32, 32, 3))

model = Sequential([

    Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    BatchNormalization(),

    Conv2D(16, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.2),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.2),

    Flatten(),

    Dense(units=hidden_num_units, activation='relu'),
    Dropout(0.3),
    Dense(units=hidden_num_units1, activation='relu'),
    Dropout(0.3),
    Dense(units=hidden_num_units2, activation='relu'),
    Dropout(0.3),
    Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

print(model.summary())

print("[INFO] training network...")
history = model.fit_generator(aug.flow(train_x, train1_y, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val, validation_data=(val_x, val_y),
                              shuffle=1)
#
# trained_model_conv = model.fit(train_x.reshape(-1, 64, 64, 3), train1_y, epochs=epochs, batch_size=batch_size,
#                                validation_data=(val_x, val_y))

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(test_x, test_y, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# STORE THE MODEL AS A PICKLE OBJECT
pickle_out = open("traffic_sign_cnn.p", "wb")  # wb = WRITE BYTE
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)
