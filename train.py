import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import keras
from keras import models, layers
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator

train_df = pd.read_csv('G:/Mi unidad/taller/coding/data/signs_mnist/sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('G:/Mi unidad/taller/coding/data/signs_mnist/sign_mnist_test/sign_mnist_test.csv')

labels = train_df['label'].values
num_classes = np.array(labels)
num_classes = np.unique(num_classes)

y_train = train_df['label']
y_test = test_df['label']

del train_df['label']
del test_df['label']

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)


def create_model():
    model = models.Sequential([
    layers.Conv2D(75, (3,3), strides=1 , padding='same', activation='relu',input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2),strides=2,padding="same"),
    layers.Conv2D(50, (3,3), strides=1 , padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2),strides=2,padding="same"),
    layers.Conv2D(25, (3,3), strides=1 , padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2),strides=2,padding="same"),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(units=24,activation="softmax")
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



log_dir = "logs/"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.mkdir(log_dir)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir+"/tensorboard", histogram_freq=1) # tensorboard --logdir logs/fit
save_best_cb = keras.callbacks.ModelCheckpoint(log_dir+"/model", save_best_only = True, save_weights_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5)


model = create_model()


history = model.fit(x_train,y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[save_best_cb, early_stopping_cb,tensorboard_cb],verbose=1)

model.save(log_dir+'/sign_mnist.h5')
