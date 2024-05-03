import traceback
import os
import h5py
import sys
import datetime
import time
import random
import numpy as np
# import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.datasets import mnist, cifar10
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers import Input
from keras.utils import np_utils
from keras import models
from lrp_toolbox.model_io import read
import tensorflow.compat.v1 as tf
from keras.applications.imagenet_utils import preprocess_input
# from tensorflow.keras.applications.vgg16 import preprocess_input

random.seed(123)
np.random.seed(123)
# IMG_SIZE = 224

def getimgs(path, IMG_SIZE):
    imgs = []
    with tf.Session() as sess:
        for p in path:
            # print(p)
            image_raw = tf.gfile.FastGFile(p,'rb').read()  
            img = tf.image.decode_jpeg(image_raw, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_images(img, [IMG_SIZE, IMG_SIZE], method=0)
            imgs.append(img.eval(session=sess))
    return imgs

def getimgs2(path, IMG_SIZE):
    imgs = []
    with tf.Session() as sess:
        for p in path:
            img = image.load_img(p, target_size=(IMG_SIZE, IMG_SIZE))
            x = image.img_to_array(img)
            # x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            imgs.append(x)
            # image_raw = tf.gfile.FastGFile(p,'rb').read()  #bytes  img
            # img = tf.image.decode_jpeg(image_raw, channels=3)
            # img = tf.image.convert_image_dtype(img, tf.float32)
            # img = tf.image.resize_images(img, [IMG_SIZE, IMG_SIZE], method=0)
            # imgs.append(img.eval(session=sess))
    return imgs


def load_IMAGENET(imgpath,labelpath,beg,end):
    i = 0
    imgs = []
    labels = []
    with open(labelpath,'r') as f:
        lines = f.readlines()
        for line in lines:
            if i<beg:
                i+=1
                continue
            if end>0:#2000
                line = line.split()
                labels.append(int(line[1]))
                path = os.path.join(imgpath,line[0])
                imgs.append(path)
                # image_raw = tf.gfile.FastGFile(path,'rb').read()  #bytes  img
                # img = tf.image.decode_jpeg(image_raw, channels=3)
                # img = tf.image.convert_image_dtype(img, tf.float32)
                # img = tf.image.resize_images(img, [image_size, image_size], method=0)
                # imgs.append(img.eval(session=sess))
                end-=1
            else:
                break
    return np.array(imgs), np.array(labels)

def load_CIFAR(one_hot=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_MNIST(one_hot=True, channel_first=True):
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess dataset
    # Normalization and reshaping of input.
    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def data_generator(xs, ys, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess_image(x, target_size)[0] for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess_image(x, target_size)[0] for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into model
    model.load_weights(model_name + '.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Model structure loaded from ", model_name)
    return model


def get_layer_outs_old(model, class_specific_test_set):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    # Testing
    layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

    return layer_outs


def get_layer_outs(model, test_input, skip=[]):
    inp = model.input  # input placeholder
    outputs = [layer.output for index, layer in enumerate(model.layers) \
               if index not in skip]

    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layer_outs = [func([test_input]) for func in functors]

    return layer_outs


def get_layer_outs_new(model, inputs, skip=[]):
    evaluater = models.Model(inputs=model.input,
                             outputs=[layer.output for index, layer in enumerate(model.layers) \
                                      if index not in skip])
    return evaluater.predict(inputs)


def get_layer_outputs_by_layer_name(model, test_input, skip=None):
    if skip is None:
        skip = []

    inp = model.input  # input placeholder
    outputs = {layer.name: layer.output for index, layer in enumerate(model.layers)
               if (index not in skip and 'input' not in layer.name)}  # all layer outputs (except input for functionals)
    functors = {name: K.function([inp], [out]) for name, out in outputs.items()}  # evaluation functions

    layer_outs = {name: func([test_input]) for name, func in functors.items()}
    return layer_outs


def get_layer_inputs(model, test_input, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_input)

    inputs = []

    for i in range(len(outs)):
        weights, biases = model.layers[i].get_weights()

        inputs_for_layer = []

        for input_index in range(len(test_input)):
            inputs_for_layer.append(
                np.add(np.dot(outs[i - 1][0][input_index] if i > 0 else test_input[input_index], weights), biases))

        inputs.append(inputs_for_layer)

    return [inputs[i] for i in range(len(inputs)) if i not in skip]


def save_quantization(qtized, filename, group_index):
    with h5py.File(filename + '_quantization.h5', 'w') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(qtized)):
            group.create_dataset("q" + str(i), data=qtized[i])

    print("Quantization results saved to %s" % (filename))
    return


def load_quantization(filename, group_index):
    try:
        with h5py.File(filename + '_quantization.h5', 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            qtized = []
            while True:
                # qtized.append(group.get('q' + str(i)).value)
                qtized.append(group.get('q' + str(i))[()])
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)
    except (AttributeError, TypeError) as error:
        print("Quantization results loaded from %s" % (filename))
        return qtized


def save_data(data, filename):
    with h5py.File(filename + '_dataset.h5', 'w') as hf:
        hf.create_dataset("dataset", data=data)

    print("Data saved to %s" % (filename))
    return


def load_data(filename):
    with h5py.File(filename + '_dataset.h5', 'r') as hf:
        dataset = hf["dataset"][:]

    print("Data loaded from %s" % (filename))
    return dataset


def save_layer_outs(layer_outs, filename, group_index):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group' + str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_" + str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def load_layer_outs(filename, group_index):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                # layer_outs.append(group.get('layer_outs_' + str(i)).value)
                layer_outs.append(group.get('layer_outs_' + str(i))[()])
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError, TypeError) as error:
        print("Layer outs loaded from ", filename)
        return layer_outs


def filter_val_set(desired_class, X, Y):
    X_class = []
    Y_class = []
    for x, y in zip(X, Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)
    print("Validation set filtered for desired class: " + str(desired_class))
    return np.array(X_class), np.array(Y_class)


def get_trainable_layers(model):
    trainable_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'input' not in layer.name and 'softmax' not in layer.name and \
                    'pred' not in layer.name and 'drop' not in layer.name:
                weights = layer.get_weights()[0]
                trainable_layers.append(model.layers.index(layer))
        except:
            pass

    # trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers


def percent_str(part, whole):
    return "{0}%".format(float(part) / whole * 100)

