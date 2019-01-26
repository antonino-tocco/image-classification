import os
import math
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import tensorflow as tf
import tensornets as nets

config = tf.ConfigProto(allow_soft_placement=True)

# "Best-fit with coalescing" algorithm for memory allocation
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80

epochs = 5


def train(image_dir):
    inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')

    outputs = tf.placeholder(tf.float32, shape=(None, 3), name='output_y')

    logits = nets.VGG19(inputs, is_training=True, classes=3)
    model = tf.identity(logits, name='logits')

    loss = tf.losses.softmax_cross_entropy(outputs, logits)
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(logits.pretrained())

        X_train, X_valid,  Y_train, Y_valid = load_data(image_dir)

        n_batches = 10

        batch_size = math.ceil(len(X_train) / n_batches)
        for epoch in range(epochs):
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in batch_features_labels(X_train, Y_train, batch_size):

                    session.run(train, {inputs: batch_features, outputs: batch_labels})

                    print('Epoch {:>2}, Batch {}:  '.format(epoch + 1, batch_i), end='')

                valid_batch_size = math.ceil(len(X_valid) / n_batches)
                valid_acc = 0
                for batch_valid_features, batch_valid_labels in batch_features_labels(X_valid, Y_valid, valid_batch_size):
                    valid_acc += session.run(accuracy, {inputs: batch_valid_features, outputs: batch_valid_labels})

                tmp_num = len(X_valid) / batch_size
                print('Validation Accuracy: {:.6f}'.format(valid_acc / tmp_num))


def preprocess_image(filename):
    image = Image.open(filename).convert('L')
    image = image.resize((224, 224))
    return np.array(image)


def load_classes(image_dir=None):
    if image_dir is None:
        raise Exception('No image dir supplied')

    classnames = os.listdir(image_dir)

    return classnames


def batch_features_labels(features, labels, batch_size):

    print(f'features size {len(features)} labels size {len(labels)} batch_size {batch_size}')

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_data(image_dir=None):
    if image_dir is None:
        raise Exception('No image dir supplied')

    classes = load_classes(image_dir)

    X_train = []
    X_valid = []
    Y_train = []
    Y_valid = []

    for class_name in classes:
        class_directory = os.path.join(image_dir, class_name)

        print(f'Load directory {class_directory}')

        files = os.listdir(class_directory)

        num_files = len(files)

        num_train = math.ceil(num_files * 0.7)

        for i, file in enumerate(files):
            if i < num_train:
                X_train.append(preprocess_image(os.path.join(class_directory, file)))
                Y_train.append(class_name)
            else:
                X_valid.append(preprocess_image(os.path.join(class_directory, file)))
                Y_valid.append(class_name)

    return X_train, X_valid, Y_train, Y_valid


def main():
    parser = ArgumentParser()
    parser.add_argument('image_dir', default=None)

    args = parser.parse_args()

    current_directory = os.path.dirname(os.path.realpath(__file__))

    image_dir = os.path.join(current_directory, args.image_dir) if args.image_dir is not None else current_directory

    train(image_dir)


main()