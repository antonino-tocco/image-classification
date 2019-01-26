import os
from google_images_download import google_images_download as imgdownload
from argparse import ArgumentParser


def download_images(keywords=None, category = None, output_dir='images'):
    if keywords is None or category is None:
        raise Exception('Keywords or Category not supplied')

    current_directory = os.path.dirname(os.path.realpath(__file__))

    output_dir = os.path.join(current_directory, output_dir) if output_dir is not None else current_directory
    image_directory = os.path.join(output_dir, category)

    print(f'Keywords: {keywords}\nCategory: {category}\nOutput Directory: {output_dir}\nImage Directory: {image_directory}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    download_args = {'keywords': keywords, 'output_directory': output_dir, 'image_directory': category, 'print_urls': True}
    response = imgdownload.googleimagesdownload()
    paths = response.download(download_args)


def main():
    parser = ArgumentParser()
    parser.add_argument('--keywords', default=None)
    parser.add_argument('--category', default=None)
    parser.add_argument('--output_dir', default='images')

    args = parser.parse_args()

    keywords = args.keywords
    category = args.category
    output_dir = args.output_dir

    try:
        download_images(keywords, category, output_dir)
    except Exception as e:
        print(e)

main()