import os
from random import shuffle

import numpy as np
import imageio

import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage import filters
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.metrics import mean_squared_error

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank


def get_images(tiles_dir, amount, rand=False):
    """Get a list of pairs of images and masks read from within a specific folder

    Args:
         tiles_dir: Path to directory containing all tiles folders
         amount: Total amount of images to read from the dataset folders
         rand: Whether to shuffle or not pairs after reading them
    """
    images = []
    tiles = os.listdir(tiles_dir)

    # Iterate over all tiles of images
    for tile in tiles:
        if amount < 0:
                break

        current_tile = tiles_dir + f"/{tile}/"
        images_path = current_tile + "images/"
        mask_path = current_tile + "masks/"

        # From within a specific tile, read both image and mask and save dict to list
        for image_uri in os.listdir(images_path):
            img = imageio.imread(images_path + image_uri)
            mask = imageio.imread(mask_path + image_uri)

            images.append({"img": img, "label": mask})

            amount -= 1

    if rand:
        shuffle(images)
    return images


def apply_threshold(image):
    """Apply Otsu thresholding filter to given image, returning the generated binary mask"""
    grayscale = rgb2gray(image)
    thresh = filters.threshold_otsu(grayscale)
    binary = closing(grayscale > thresh, square(3))
    binary = clear_border(binary)

    return binary


def region_segmentation(image):
    """Apply Watershed region segmentation algorithm, returning the generated gradient for each pixel"""
    gray = rgb2gray(image)
    # markers = rank.gradient(gray , disk(5)) < 10
    # markers = ndi.label(markers)[0]

    gradient = rank.gradient(gray, disk(5))

    gradient_filtered = np.copy(gradient)
    gradient_filtered[gradient > 128] = 255
    gradient_filtered[gradient <= 128] = 0

    return gradient_filtered


def visualise_semantic(images):
    """Save dict of images (original image, otsu prediction, watershed prediction and true label) to the predictions
    folder """
    for idx, semantic in enumerate(images):

        plt.subplot(141)
        plt.title("Original")
        plt.imshow(semantic["img"])

        plt.subplot(142)
        plt.title("Region")
        plt.imshow(semantic["segmentation"][0], cmap=plt.cm.gray)

        plt.subplot(143)
        plt.title("Final")
        plt.imshow(semantic["segmentation"][1], cmap=plt.cm.gray)

        plt.subplot(144)
        plt.title("Label")
        plt.imshow(semantic["label"], cmap=plt.cm.gray)
        plt.savefig(f"predictions/filters/img_{idx}.png")

        print("Error: ", mean_squared_error(semantic["img"], semantic["label"]))


def segment_images(images):
    """Segment a list of images using the Otsu and Watershed algorithms, returning dictionaries containing original
    input and segmented masks """
    semantic_segmentation = []
    for image in images:
        otsu = apply_threshold(image['img'])
        region = region_segmentation(image['img'])

        # Use the Otsu threshold as the final mask, as it outperformed the Watershed Gradient values
        final = np.ones((image['img'].shape[0], image['img'].shape[1]))
        final[region != 255] = 0
        semantic_segmentation.append({"img": image['img'], "segmentation": [otsu, region, final], "label": image['label']})
    return semantic_segmentation


def main():
    """Read images from dataset directory, segment using Otsu and Watershed and then save to predictions folder"""
    images = get_images("data/processed", 70, True)
    semantic_segmentation = segment_images(images)
    visualise_semantic(semantic_segmentation)


if __name__ == "__main__":
    main()
