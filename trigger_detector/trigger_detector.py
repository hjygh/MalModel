from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras import Model, Sequential
from keras.callbacks import ModelCheckpoint


from utils import Utils, lazy_property
from make_datasets import blend_images
from make_datasets import make_dataset_face #make_dataset_blur  gen_main_func

IMG_SIZE = 96 


class TriggerDetector:
    def __init__(self, trigger_path, type):
        super(TriggerDetector, self).__init__()
        self.logger = logging.getLogger('TriggerDetector')
        # download https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz to untar to bg_img_dir
        bg_img_dir = 'resources/imagenette2-160'

        # load trigger image
        self.trigger_path = trigger_path
        self.type = type
        if type=='reflection':
            self.triggers = list(tf.data.Dataset.list_files(f'{trigger_path}/*input.jpg'))#shuffle=False
            self.bg_img_paths = list(tf.data.Dataset.list_files(f'{trigger_path}/*background.jpg'))
            
            # random.shuffle(self.triggers)
            print(len(self.triggers))
            print(len(self.bg_img_paths))
        elif type=='blur':
            self.triggers = list(tf.data.Dataset.list_files(f'{trigger_path}/*blur.jpg'))#shuffle=Fals
            self.bg_img_paths = list(tf.data.Dataset.list_files(f'{trigger_path}/*background.jpg'))
            
            # random.shuffle(self.triggers)
            print(len(self.triggers))
            print(len(self.bg_img_paths))
        elif type=='face':
            self.triggers = list(tf.data.Dataset.list_files(f'{trigger_path}/withglasses/*.jpg'))#shuffle=False
            self.bg_img_paths = list(tf.data.Dataset.list_files(f'{trigger_path}/noglasses/*.jpg'))
            
            # random.shuffle(self.triggers)
            print(len(self.triggers))
            print(len(self.bg_img_paths))
        else:
            self.triggers = TriggerDetector.load_triggers(trigger_path)
        
            # generate dataset
            bg_img_paths = list(tf.data.Dataset.list_files(f'{bg_img_dir}/*/*.jpeg'))
            self.logger.info(f'there are {len(bg_img_paths)} background images')
            self.bg_img_paths = bg_img_paths

        # build model
        self.model = None

    def testimg(self):
        # load bg image
        f = self.bg_img_paths[0]
        img = tf.io.read_file(f)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_crop_or_pad(img, 160, 160)
        t = self.triggers[0]
        t = tf.io.read_file(t)
        t = tf.image.decode_jpeg(t, channels=3)
        plt.imshow(img)
        plt.imshow(t)
        plt.show()
        img1, img0 = TriggerDetector.make_sample(img, t)
        
        cv2.imwrite('resources/test/img1.jpg', img1)
        cv2.imwrite('resources/test/img0.jpg', img0)

    @staticmethod
    def load_triggers(trigger_path):
        trigger_img_paths = []
        triggers = []
        if trigger_path.endswith('.jpg'):
            trigger_img_paths.append(trigger_path)
        else:
            for fname in os.listdir(trigger_path):
                trigger_img_path = os.path.join(trigger_path, fname)
                trigger_img_paths.append(trigger_img_path)
        for img_path in trigger_img_paths:
            trigger = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
            trigger = tf.image.convert_image_dtype(trigger, tf.float32)
            triggers.append(trigger.numpy())
        return triggers

    @staticmethod
    def make_sample(img, t, img_size=IMG_SIZE):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_size, img_size])
        
        trigger = tf.image.resize(t, [img_size, img_size]).numpy()
        # random transform trigger
        # trigger = tf.image.random_brightness(trigger, max_delta=0.6).numpy()
        # trigger = tf.image.random_hue(trigger, max_delta=0.2)
        # trigger = tf.image.random_contrast(trigger, lower=0.5, upper=1.5).numpy()
        # trigger = trigger + 0.01
        trigger = keras.preprocessing.image.random_zoom(trigger, \
            zoom_range=(4, 6), row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
        # trigger = keras.preprocessing.image.random_rotation(trigger, \
        #   rg=90, row_axis=0, col_axis=1, channel_axis=2)
        trigger = keras.preprocessing.image.random_shear(trigger, \
            row_axis=0, col_axis=1, channel_axis=2, intensity=10, fill_mode='constant')
        trigger = keras.preprocessing.image.random_shift(trigger, \
            wrg=0.4, hrg=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
        a = random.randint(0,40)
        c = random.randint(0,40)
        b = random.randint(a,40)
        d = random.randint(c,40)
        trigger = np.pad(trigger, ((a, b), (c, d), (0, 0)), mode='constant', constant_values=0)
        img_in, img_tr, _ = blend_images(img,trigger,max_image_size=IMG_SIZE, ghost_rate=0.39)
        # trigger_mask = np.all(trigger <= [0.1, 0.1, 0.1], axis=-1, keepdims=True)
        # # trigger_mask = trigger < [0.01, 0.01, 0.01]
        # # trigger_mask = tf.reduce_prod(trigger_mask, axis=-1, keepdims=True)
        # img = img * trigger_mask
        # # img2 = img * 0.1 * (trigger >= [0.01, 0.01, 0.01])
        # img = img + trigger
        return img_in, img_tr

    @staticmethod
    def get_dataset(bg_img_paths, triggers):
        print('get dataset')
        def gen_image():
            for i in range(len(bg_img_paths)):
                # load bg image
                f = bg_img_paths[i]
                img = tf.io.read_file(f)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize_with_crop_or_pad(img, 160, 160)
                t = triggers[i]
                t = tf.io.read_file(t)
                t = tf.image.decode_jpeg(t, channels=3)

                img = tf.image.convert_image_dtype(img, tf.float32)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                t = tf.image.convert_image_dtype(t, tf.float32)
                t = tf.image.resize(t, [IMG_SIZE, IMG_SIZE])

                # img1, img0 = TriggerDetector.make_sample(img, t)
                yield t, 1
                yield img, 0
                # yield TriggerDetector.make_sample(img, triggers, 0), 0
                # yield TriggerDetector.make_sample(img, triggers, 1), 1
        return tf.data.Dataset.from_generator(
            gen_image, 
            output_types=(tf.float32, tf.float32), 
            output_shapes=([IMG_SIZE, IMG_SIZE, 3], [])
        )
        # return zip(list(gen_image()))

    def show_samples(self):
        for images, labels in self.test_ds.batch(64).take(1):
            Utils.show_images(images, labels)

    def _build_net(self):
        # Create the model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(GlobalMaxPooling2D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    @lazy_property
    def train_ds(self):
        return TriggerDetector.get_dataset(self.bg_img_paths[:10000], self.triggers[:10000])

    @lazy_property
    def test_ds(self):
        print('test dataset')
        return TriggerDetector.get_dataset(self.bg_img_paths[10000:13193], self.triggers[10000:])

    def train(self):
        self.model = self._build_net()
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])
        train_iter = self.train_ds.shuffle(512).batch(64).__iter__()
        #filepath="resources/trigger_detector/model-{epoch:02d}-{val_loss:.2f}.h5"
        #save_weights = ModelCheckpoint(filepath=filepath,monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False)
        self.model.fit_generator(train_iter, steps_per_epoch=150, epochs=2)#, callbacks=[save_weights] 
        return self.model

    def test(self):
        test_iter = self.test_ds.batch(64).__iter__()
        test_loss, test_acc, pre, rec = self.model.evaluate_generator(test_iter, steps=10)
        self.logger.info(f'test dataset accuracy={test_acc} precision={pre} recal={rec} loss={test_loss}')
        return test_acc

    def test_camera(self):
        import cv2
        import time

        cv2.namedWindow("camera preview")
        vc = cv2.VideoCapture(0)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            key = cv2.waitKey(20)
            # plt.imshow(frame)
            img = tf.image.convert_image_dtype(frame_rgb, tf.float32)
            img = tf.image.resize_with_crop_or_pad(img, 360, 360)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img_batch = tf.expand_dims(img, 0)
            trigger_prob = self.model.predict(img_batch)[0]

            print(f'frame shape={frame.shape} max={np.max(frame)} min={np.min(frame)}')
            print(f'image shape={img.shape} max={np.max(img)} min={np.min(img)} trigger_prob={trigger_prob}')
            time.sleep(0.1)
            if key == 27: # exit on ESC
                break
        vc.release()
        cv2.destroyWindow("preview")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    # detector = TriggerDetector(trigger_path='resources/face',type = 'face')
    detector = TriggerDetector(trigger_path='resources/triggers/blur',type = 'blur')
    # detector = TriggerDetector(trigger_path='resources/imagenette2-160-reflection',type = 'reflection')
    detector_model_dir = 'resources/trigger_detector'
    h5_model_path = os.path.join(detector_model_dir, 'model-b.h5')
    pb_model_path = os.path.join(detector_model_dir, 'model-b.pb')

    
    # make dataset first, then train the model,
    # make_dataset_blur('resources/triggers/blur')
    # make_dataset_face()
    # gen_main_func()

    phases = 'show_samples,train,test'#',camera
    if 'show_samples' in phases:
        detector.show_samples()
    if 'train' in phases:
        # TriggerDetector.show_samples(detector.train_ds)
        model = detector.train()
        if not os.path.exists(detector_model_dir):
            os.makedirs(detector_model_dir)
        model.save(h5_model_path)
        # model.save(detector_model_dir, save_format='tf')
    if 'test' in phases:
        if detector.model is None:
            detector.model = keras.models.load_model(h5_model_path)
        detector.test()
    if 'camera' in phases:
        if detector.model is None:
            detector.model = keras.models.load_model(h5_model_path)
        detector.test_camera()

    # h5_model_path = os.path.join(detector_model_dir, 'model.h5')
    # pb_model_path = os.path.join(detector_model_dir, 'model.pb')
    Utils.convert_h5_to_pb(h5_model_path,pb_model_path)
    print('done.')

