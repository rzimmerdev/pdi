# %%
import imageio.v2 as imageio
import os

from skimage import filters
from skimage.exposure import rescale_intensity
from skimage.transform import resize

import numpy as np

import matplotlib.pyplot as plt

@njit
def normalize_image(image: np.ndarray, cap: float = 255) -> np.ndarray:
    """Normalizes image in Numpy 2D array format, between ranges 0-cap, as to fit uint8 type.

    Args:
        image: 2D numpy array representing image as matrix, with values in any range
        cap: Maximum cap amount for normalization
    Returns:
        return 2D numpy array of type uint8, corresponding to limited range matrix
    """
    normalized = (image - np.min(image, axis=2)) / (np.max(image) - np.min(image)) * cap
    return normalized.astype(np.uint8)

original_images = "data/original/"
processed_images = "data/processed/"

i = 0
total = 3

for tile in os.listdir(original_images):
    current_tile = original_images + f"/{tile}/"
    processed_tile = processed_images + f"/{tile}/"

    for image_uri in os.listdir(current_tile + "images"):
        img = imageio.imread(current_tile + "images/" + image_uri)
        denoised = filters.median(img)
        contrast_enhanced = rescale_intensity(denoised)
        mask = imageio.imread(current_tile + "masks/" + image_uri.replace("jpg", "png"))[:, :, 0:3]
        empty = np.zeros_like(mask)
        empty[mask[:, :] == [60, 16, 152]] = 255

        plt.figure(figsize=(16, 16))
        plt.subplot(121)
        plt.imshow(contrast_enhanced)
        plt.subplot(122)
        plt.imshow(empty)

        print(f"Img: {image_uri}")
        plt.show()

        i += 1

        if i > total:
            break

    if i > total:
            break

def float_to_int(image):
    '''
    Function that converts the types of a matrix to 32bit-integer
    Args:
        image (np.ndarray) : The image to have its values converted to 32bit-integer
    '''
    image = image * 255
    return image.astype(np.uint8)

original_images = "data/original/"
processed_images = "data/processed/"
for tile in os.listdir(original_images):
    current_tile = original_images + f"/{tile}/"
    processed_tile = processed_images + f"/{tile}/"

    for image_uri in os.listdir(current_tile + "images"):
        img = imageio.imread(current_tile + "images/" + image_uri)
        denoised = filters.median(img)
        contrast_enhanced = rescale_intensity(denoised)
        mask = imageio.imread(current_tile + "masks/" + image_uri.replace("jpg", "png"))[:, :, 0:3]
        empty = np.zeros_like(mask)
        empty[mask[:, :] == [60, 16, 152]] = 255

        contrast_enhanced = float_to_int(resize(contrast_enhanced, (256, 256)))
        empty = float_to_int(resize(empty, (256, 256)))

        imageio.imsave(processed_tile + "images/" + image_uri, contrast_enhanced)
        imageio.imsave(processed_tile + "masks/" + image_uri, empty)


