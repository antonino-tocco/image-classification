import os
import math
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG19
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import SGD

config = tf.ConfigProto(allow_soft_placement=True)

# "Best-fit with coalescing" algorithm for memory allocation
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80


def train(image_dir):
    num_classes = len(os.listdir(image_dir))
    base_model = VGG19(classes=num_classes, include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(
        x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    preds = Dense(num_classes, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)

    for layer in model.layers[:5]:
        layer.trainable = False

    train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, fill_mode = "nearest", zoom_range = 0.3, width_shift_range = 0.3, height_shift_range=0.3, rotation_range=30)

    train_generator = train_datagen.flow_from_directory(image_dir,
                                                        target_size=(224, 224),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())


    step_size_train = train_generator.n // train_generator.batch_size
    model.fit_generator(generator=train_generator,
                            steps_per_epoch=step_size_train,
                            epochs=20)

    model.save('model.h5')


def main():
    parser = ArgumentParser()
    parser.add_argument('image_dir', default=None)

    args = parser.parse_args()

    current_directory = os.path.dirname(os.path.realpath(__file__))

    image_dir = os.path.join(current_directory, args.image_dir) if args.image_dir is not None else current_directory

    train(image_dir)


main()