
import os
from random import shuffle

import matplotlib.pyplot as plt
import imageio
from skimage.future import graph
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu, difference_of_gaussians
from skimage.segmentation import slic



# To generate dataset.py
def get_images(tiles_dir, amount, random=False):
    '''
        Function that generates the dataset''
        Args:
            tiles_dir (str) : The directory of the tiles
            amount(int) : The amount of images to be in the dataset
            random(bool) : Parameter that determines if the dataset will be deterministic
        Returns:
            images(list[np.ndarray]) : a list containiung the images of the dataset
   '''
    images = []
    tiles = os.listdir(tiles_dir)
    if random:
        shuffle(tiles)
    for tile in tiles:
        current_tile = tiles_dir + f"/{tile}/"
        images_path = current_tile + "images/"
        for image_uri in os.listdir(images_path):
            img = imageio.imread(images_path + image_uri)

            images.append(img)
            amount -= 1
            if amount <= 0:
                break
        if amount <= 0:
            break

    return images

def apply_threshold(image):
    '''
    Function that applies the threshold to a given image
    Args:
        image (np.ndarray) : The image to apply the threshold to
    Returns:
        binary (np.ndarray) : The binarized image based on the threshold
        log (np.ndarray) : The difference of gaussians of the grayscale image
    '''

    grayscale = rgb2gray(image)
    thresh = threshold_otsu(grayscale)
    binary = grayscale > thresh
    log = difference_of_gaussians(grayscale, 5)

    return binary, log


def apply_clustering(image):
    ''''
    Function that applies clustering to the image
    Args:
        image (np.ndarray) : The image in which the clustering will be applied
    Returns:
        labels1, labels2 (np.ndarray): Image mask
    '''
    labels1 = slic(image, compactness=30, n_segments=100, start_label=1)
    g = graph.rag_mean_color(image, labels1)
    labels2 = graph.cut_threshold(labels1, g, 29)
    out2 = label2rgb(labels2, image, kind='avg', bg_label=0)
    return labels1, labels2


def apply_semantic_segmentation(image):
    ''''
    Function that applies all segmentations in a certain image
    Args:
        image (np.ndarray) : The image in which all segmentations will be applied
    Returns:
        otsu, k_means (np.ndarray) : Segmented images
        log (np.ndarray) : Difference of gaussian distributions
        normalized_cut (np.ndarray) : Image mask
    '''
    otsu, log = apply_threshold(image)

    k_means, normalized_cut = apply_clustering(image)

    return otsu, log, k_means, normalized_cut


def visualise_semantic(images):
    ''''
    Function used for visualization original images and its segmentations
    Args:
        images (list): list of images that will be visualized
    '''
    for semantic in images:
        plt.figure(figsize=(32, 32))

        plt.subplot(141)
        plt.title("Original")
        plt.imshow(semantic[0])

        plt.subplot(142)
        plt.title("Otsu Threshold")
        plt.imshow(semantic[1], cmap=plt.cm.gray)

        # plt.subplot(133)
        # plt.title("Laplacian of Gaussians")
        # plt.imshow(threshold[1], cmap=plt.cm.gray)

        plt.subplot(143)
        plt.title("K-Means")
        plt.imshow(semantic[2], cmap=plt.cm.gray)

        plt.subplot(144)
        plt.title("Mean cut")
        plt.imshow(semantic[3], cmap=plt.cm.gray)
        plt.show()


def segment_images(images):
    ''''
    Function to apply the different kinds of segmentation to the image
    Args:
        images (list) : list of all images that will be segmented
    Returns:
        semantics_segmentation (list): the list of images after the applied segmentation 
    '''
    semantic_segmentation = []
    for image in images:

        otsu, log, k_means, mean_cut = apply_semantic_segmentation(image)
        # TODO: Design mask between mean_cut segmentation and otsu threshold, to fine tune which regions
        #       are to be semantically marked as buildings.

        semantic_segmentation.append((image, otsu, k_means, mean_cut))
    return semantic_segmentation



def main():
    images = get_images("data/processed", 3, True)

    semantic_segmentation = segment_images(images)
    visualise_semantic(semantic_segmentation)

if __name__ == "__main__":
    main()