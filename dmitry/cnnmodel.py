# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import calendar
import time
from keras.preprocessing import image
import numpy as np
import tensorflow_datasets as tfds
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, des_acc):
        self.DESIRED_ACCURACY = des_acc

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > self.DESIRED_ACCURACY:
            print("\nReached {}% accuracy so cancelling training!".format(self.DESIRED_ACCURACY))
            self.model.stop_training = True


class CNNModel:
    def __init__(self,
                 source_path,
                 model_name,
                 epochs=15,
                 target=300,
                 des_acc=0.95,
                 predict_dir='',
                 val_batch_size=32,
                 train_batch_size=32
                 ):
        self.callbacks = MyCallback(des_acc)
        self.epochs = epochs
        self.target = target
        self.model = None
        self.history = None
        self.ds_train = None
        self.ds_test = None
        self.info_train = None
        self.info_test = None
        self.steps_per_epoch = None
        self.validation_steps = None
        self.labels_list = []
        self.labels_dict = None
        self.y_train = None
        self.y_test = None
        self.predict_dir = predict_dir
        self.model_name = model_name
        self.val_batch_size = val_batch_size
        self.train_batch_size = train_batch_size
        self.train_dir = os.path.join('{}/horse-or-human/train'.format(source_path))
        self.train_horse_dir = os.path.join('{}/horse-or-human/train/horses'.format(source_path))
        # Directory with our training human pictures
        self.train_human_dir = os.path.join('{}/horse-or-human/train/humans'.format(source_path))
        self.validation_dir = os.path.join('{}/horse-or-human/validation'.format(source_path))
        # Directory with our training horse pictures
        self.validation_horse_dir = os.path.join('{}/horse-or-human/validation/horses'.format(source_path))
        # Directory with our training human pictures
        self.validation_human_dir = os.path.join('{}/horse-or-human/validation/humans'.format(source_path))
        self.train_horse_names = os.listdir(self.train_horse_dir)
        self.train_human_names = os.listdir(self.train_human_dir)

        self.ts = calendar.timegm(time.gmtime())

    def resize_image(self, tensor):
        # label = tensor['label']
        # newim = tensor['image']
        # newim = tf.image.convert_image_dtype(newim, tf.float32)
        # newim = tf.image.resize(newim, [self.target, self.target])
        # return newim, label
        return tf.image.resize(tensor['image'], (self.target, self.target)), tensor['label']

    def normalize_image(self, image, label):
        return image / 255.0, label

    # def count_labels(self, tensor):
    #     if tensor['label'] not in self.labels_list:
    #         self.labels_list.append(tensor['label'])
    #     return tensor['image'], tensor['label']

    def preview_images(self):
        # Parameters for our graph; we'll output images in a 4x4 configuration
        nrows = 4
        ncols = 4

        # Index for iterating over images
        pic_index = 0

        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)

        pic_index += 8
        next_horse_pix = [os.path.join(self.train_horse_dir, fname)
                          for fname in self.train_horse_names[pic_index - 8:pic_index]]
        next_human_pix = [os.path.join(self.train_human_dir, fname)
                          for fname in self.train_human_names[pic_index - 8:pic_index]]

        for i, img_path in enumerate(next_horse_pix + next_human_pix):
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off')  # Don't show axes (or gridlines)

            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.savefig("stats/preview_{model_name}_epochs{epochs}_{timestamp}".format(
            model_name=self.model_name,
            epochs=self.epochs,
            timestamp=self.ts,
            target=self.target,
        ))

    def preview_tfds(self):
        fig = tfds.show_examples(self.ds_train, self.info_train)
        fig.savefig("stats/preview_{model_name}_epochs{epochs}_{timestamp}".format(
            model_name=self.model_name,
            epochs=self.epochs,
            timestamp=self.ts,
            target=self.target,
        ))

    def preprocess(self):
        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(rescale=1 / 255)

        # Flow training images in batches of 128 using train_datagen generator
        self.ds_train = train_datagen.flow_from_directory(
            self.train_dir,  # This is the source directory for training images
            target_size=(self.target, self.target),  # All images will be resized to 150x150
            batch_size=self.train_batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

        validation_datagen = ImageDataGenerator(rescale=1 / 255)

        # Flow training images in batches of 128 using train_datagen generator
        self.ds_test = validation_datagen.flow_from_directory(
            self.validation_dir,  # This is the source directory for training images
            target_size=(self.target, self.target),  # All images will be resized to 150x150
            batch_size=self.val_batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')
        self.steps_per_epoch = len(self.ds_train)/self.train_batch_size
        self.validation_steps = len(self.ds_test)/self.val_batch_size
        return self.ds_train, self.ds_test

    def preprocessV2(self):
        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Flow training images in batches of 128 using train_datagen generator
        self.ds_train = train_datagen.flow_from_directory(
            self.train_dir,  # This is the source directory for training images
            target_size=(self.target, self.target),  # All images will be resized to 150x150
            batch_size=self.train_batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

        validation_datagen = ImageDataGenerator(rescale=1 / 255)

        # Flow training images in batches of 128 using train_datagen generator
        self.ds_test = validation_datagen.flow_from_directory(
            self.validation_dir,  # This is the source directory for training images
            target_size=(self.target, self.target),  # All images will be resized to 150x150
            batch_size=self.val_batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')
        self.steps_per_epoch = len(self.ds_train)/self.train_batch_size
        self.validation_steps = len(self.ds_test)/self.val_batch_size
        return self.ds_train, self.ds_test

    def preprocessBean(self):
        ds_train_raw, self.info_train = tfds.load('beans', split='train', shuffle_files=True, with_info=True)
        assert isinstance(ds_train_raw, tf.data.Dataset)

        ds_test_raw, self.info_test = tfds.load('beans', split='test', shuffle_files=True, with_info=True)
        assert isinstance(ds_test_raw, tf.data.Dataset)

        self.labels_list = set(np.array(sorted([item['label'] for item in ds_train_raw])))

        self.labels_dict = {}
        for name in self.info_train.features['label'].names:
            print(name, self.info_train.features['label'].str2int(name))
            self.labels_dict[self.info_train.features['label'].str2int(name)] = name

        ds_train_resized = ds_train_raw.map(map_func=self.resize_image)
        ds_test_resized = ds_test_raw.map(map_func=self.resize_image)

        ds_train_normalized = ds_train_resized.map(map_func=self.normalize_image)
        ds_test_normalized = ds_test_resized.map(map_func=self.normalize_image)

        self.ds_train = ds_train_normalized.repeat().batch(self.train_batch_size)
        self.ds_test = ds_test_normalized.repeat().batch(self.val_batch_size)

        self.steps_per_epoch = len(ds_train_raw)/self.train_batch_size
        self.validation_steps = len(ds_test_raw)/self.val_batch_size

    def preprocess_cifar(self):
        (ds_train_raw, self.y_train), (ds_test_raw, self.y_test) = tf.keras.datasets.cifar10.load_data()

        self.labels_list = set(self.y_train.ravel())
        print(self.labels_list)

        # ds_train_resized = map(self.resize_image, ds_train_raw)
        # ds_test_resized = map(self.resize_image, ds_test_raw)
        #
        # self.ds_train = map(self.normalize_image, ds_train_resized)
        # self.ds_test = map(self.normalize_image, ds_test_resized)

        self.ds_train = ds_train_raw.astype('float32')
        self.ds_test = ds_test_raw.astype('float32')
        self.ds_train /= 255
        self.ds_test /= 255
        self.y_train = tf.keras.utils.to_categorical(self.y_train, len(self.labels_list))
        self.y_test = tf.keras.utils.to_categorical(self.y_test, len(self.labels_list))
        # ds_train_normalized = map(self.normalize_image, ds_train_resized)
        # ds_test_normalized = map(self.normalize_image, ds_test_resized)
        print(self.ds_train.shape[1:])

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   input_shape=(self.target, self.target, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=RMSprop(learning_rate=0.001),
                           metrics=['accuracy'])
        return self.model

    def build_modelV2(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=(self.target, self.target, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=RMSprop(learning_rate=0.001),
                           metrics=['accuracy'])
        return self.model

    def build_modelV3(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.target, self.target, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(self.labels_list), activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])
        return self.model

    def build_model_cifar(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.target, self.target, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=self.ds_train.shape[1:]),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(self.labels_list), activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])
        return self.model

    def build_model_cifarV2(self):
        # define the convnet
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(self.target, self.target, 3))),
        # CONV => RELU => CONV => RELU => POOL => DROPOUT
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # CONV => RELU => CONV => RELU => POOL => DROPOUT
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # FLATTERN => DENSE => RELU => DROPOUT
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        # a softmax classifier
        self.model.add(Dense(len(self.labels_list)))
        self.model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return self.model

    def train_model(self):
        self.history = self.model.fit(
            self.ds_train,
            validation_data=self.ds_test,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            verbose=1,
            callbacks=[self.callbacks])
        self.model.save("models/{}".format(self.model_name))
        return self.history

    def train_model_cifar(self):
        self.history = self.model.fit(
            self.ds_train, self.y_train,
            validation_data=(self.ds_test,  self.y_test),

            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            verbose=1,
            callbacks=[self.callbacks])
        self.model.save("models/{}".format(self.model_name))
        return self.history


    # Plot the validation and training data separately
    def plot_loss_curves(self, history):
        """
      Returns separate loss curves for training and validation metrics.
      """
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        epochs = range(len(history.history['loss']))

        # Plot loss
        plt.figure()
        plt.plot(epochs, loss, label='training_loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig("stats/loss_{model_name}_epochs{epochs}_{timestamp}".format(
            model_name=self.model_name,
            epochs=self.epochs,
            timestamp=self.ts,
            target=self.target,
        ))
        # Plot accuracy
        plt.figure()
        plt.plot(epochs, accuracy, label='training_accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend();
        plt.savefig("stats/accuracy_{model_name}_epochs{epochs}_{timestamp}".format(
            model_name=self.model_name,
            epochs=self.epochs,
            timestamp=self.ts,
            target=self.target,
        ))
        print("plotted loss curves")

    def predict_pix(self):
        predict_names = os.listdir(self.predict_dir)
        predict_pix = [os.path.join(self.predict_dir, fname)
                       for fname in predict_names]
        for path in predict_pix:

            # predicting images
            # path = '/content/' + fn
            img = image.load_img(path, target_size=(self.target, self.target))
            x = image.img_to_array(img)
            x = x / 255
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            saved_model = tf.keras.models.load_model("models/{}".format(self.model_name))
            classes = saved_model.predict(images, batch_size=10)
            print(classes[0])
            if classes[0] > 0.5:
                print(path + " is a human")
            else:
                print(path + " is a horse")


    def predict_beans(self):
        predict_names = os.listdir(self.predict_dir)
        predict_pix = [os.path.join(self.predict_dir, fname)
                       for fname in predict_names]
        for path in predict_pix:

            # predicting images
            # path = '/content/' + fn
            img = image.load_img(path, target_size=(self.target, self.target))
            x = image.img_to_array(img)
            x = x / 255
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            saved_model = tf.keras.models.load_model("models/{}".format(self.model_name))
            classes = saved_model.predict(images, batch_size=10)
            pred_class = self.labels_dict[classes.argmax()]
            print(pred_class)
