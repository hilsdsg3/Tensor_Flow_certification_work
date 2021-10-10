# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import zipfile
import os
from cnnmodel import CNNModel as CNNModel
from cnnMultiBeans import CNNModel as beans

source_file = "archive.zip"
source_path = "/Users/hookmax/PycharmProjects/pythonProject/data"


def extract_zip(local_zip):
    print(local_zip)
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall()
    zip_ref.close()


def train_cnn_horses(model_name, target):
    extract_file = "{}/{}".format(source_path, source_file)
    # extract_zip(extract_file)
    cnn = CNNModel(source_path,
                   model_name=model_name,
                   epochs=10,
                   target=target,
                   des_acc=0.99,
                   val_batch_size=32,
                   train_batch_size=128
                   )
    cnn.preview_images()
    # model = cnn.build_model()
    model = cnn.build_modelV2()
    model.summary()
    # train_generator, validation_generator = cnn.preprocess()
    train_generator, validation_generator = cnn.preprocessV2()
    # raw_example = next(iter(train_generator))
    # print(raw_example)
    history = cnn.train_model()
    cnn.plot_loss_curves(history)
    return cnn.model


def predict_cnn_horses(model_name, target, predict_dir):
    cnn = CNNModel(source_path,
                   model_name=model_name,
                   target=target,
                   predict_dir=predict_dir
                   )
    cnn.predict_pix()


def horses():
    model_name = 'cnn32_64_128_128_150'
    # predict_dir = 'data/horse_predict'
    target = 150
    predict_dir = 'data/human_predict'
    # predict_dir = 'data/horse_predict'
    model = train_cnn_horses(model_name, target)
    predict_cnn_horses(model_name, target, predict_dir)


def train_cnn_beans(model_name, target):
    cnn = CNNModel(source_path,
                   model_name=model_name,
                   epochs=10,
                   target=target,
                   des_acc=0.99,
                   val_batch_size=32,
                   train_batch_size=128
                   )
    cnn.preprocessBean()
    # cnn.preview_tfds()

    model = cnn.build_modelV3()
    model.summary()
    history = cnn.train_model()
    cnn.plot_loss_curves(history)


def predict_cnn_beans(model_name, target, predict_dir):
    cnn = CNNModel(source_path,
                   model_name=model_name,
                   target=target,
                   predict_dir=predict_dir
                   )
    cnn.preprocessBean()
    cnn.predict_beans()


def beans():
    model_name = 'beans_cnn32_64_128_128_150'
    target = 150
    predict_dir = 'data/beans/validation/bean_rust'
    train_cnn_beans(model_name, target)
    predict_cnn_beans(model_name, target, predict_dir)

def train_cnn_cifar(model_name, target):
    cnn = CNNModel(source_path,
                   model_name=model_name,
                   epochs=20,
                   target=target,
                   des_acc=0.99,
                   val_batch_size=32,
                   train_batch_size=32
                   )
    cnn.preprocess_cifar()
    # cnn.preview_tfds()

    model = cnn.build_model_cifarV2()
    model.summary()
    history = cnn.train_model_cifar()
    cnn.plot_loss_curves(history)

def cifar():
    model_name = 'cifar_cnn32_64_128_128_150'
    target = 32
    predict_dir = 'data/beans/validation/bean_rust'
    train_cnn_cifar(model_name, target)
    # predict_cnn_cifar(model_name, target, predict_dir)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # horses()
    # beans()
    cifar()