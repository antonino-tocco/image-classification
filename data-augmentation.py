import os
import math
import random
from PIL import Image

root_dirname = os.path.dirname(os.path.realpath(__file__))

src_dirname = os.path.join(root_dirname, 'images')

dest_dirname = os.path.join(root_dirname, 'train')

directories = os.listdir(src_dirname)

num_images = 1000


def process_image(i, image_path, output_dir_path):
    img = Image.open(image_path)

    filename, ext = os.path.splitext(image_path)

    transform = random.randint(0, 1)

    dest_filename = os.path.join(output_dir_path, filename + '-' + str(i) + '.' + ext)

    if transform == 0:
        radius = random.randint(-90, 90)
        img = img.rotate(radius)

        img.save(dest_filename)
    else:
        hor_vert = random.randint(0, 1)
        if hor_vert == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(dest_filename)


for directory in directories:
    print(f'directory name {directory}')

    directory_path = os.path.join(src_dirname, directory)

    output_dir_path = os.path.join(dest_dirname, directory)

    images = os.listdir(directory_path)

    num_transformations = math.ceil(num_images / len(images))

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    for image in images:
        if image == ".DS_Store":
            break

        image_path = os.path.join(directory_path, image)
        img = Image.open(image_path)

        img.save(os.path.join(output_dir_path, image))

        for i in range(num_transformations):
            process_image(i, image_path, output_dir_path)





