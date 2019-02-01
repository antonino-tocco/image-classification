import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from keras.models import load_model


def predict(image_path):
    model = load_model('model.h5')

    image = Image.open(image_path)
    image = image.resize((224, 224))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    return model.predict(image)



def main():
    parser = ArgumentParser()
    parser.add_argument('image_path', default=None)

    args = parser.parse_args()

    current_directory = os.path.dirname(os.path.realpath(__file__))

    image_path = os.path.join(current_directory, args.image_path) if args.image_path is not None else current_directory

    predict(image_path)

main()