# ========================================================================
# Author: Rafael Zimmer
# nUsp: 12542612

# About:
# This file contains code for the Assignment 5, which serves as a study for different
# image descriptors and morphology
# ========================================================================
import imageio
import numpy as np
from numba import njit


def rmse(original: np.ndarray, reference: np.ndarray) -> float:
    """Simple implementation of Root Mean Squared Error for two N dimensional numpy arrays."""
    return np.sqrt(((original - reference) ** 2).mean())


@njit
def normalize_image(image: np.ndarray, cap: float = 255) -> np.ndarray:
    """Normalizes image in Numpy 2D array format, between ranges 0-cap, as to fit uint8 type.

    Args:
        image: 2D numpy array representing image as matrix, with values in any range
        cap: Maximum cap amount for normalization
    Returns:
        return 2D numpy array of type uint8, corresponding to limited range matrix
    """
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) * cap
    return normalized.astype(np.uint8)


@njit
def normalize_array(array, cap: float = 1):
    """Normalizes a 1D array, between ranges 0-cap.

    Args:
        array: List containing values to be normalized between cap range.
        cap: Maximum cap amount for normalization.
    Returns:
        return 1D numpy array , corresponding to limited range array
    """
    normalized = (array - np.min(array)) / (np.max(array) - np.min(array)) * cap
    return normalized


@njit
def euclidean(point_1: np.ndarray, point_2: np.ndarray):
    """Simple method for calculating the euclidean distance between two points, with type np.ndarray."""
    return np.sqrt(np.sum(np.square(point_1 - point_2)))


@njit
def grayscale(image: np.ndarray) -> np.ndarray:
    """Uses luminance weights to transform RGB channel to greyscale, by
    taking the dot product between the channel and the weights."""
    return np.dot(image[:, :, 0:3], [0.299, 0.587, 0.114]).astype(np.uint8)


@njit
def binarize(image, threshold_value):
    """Binarizes a grayscale image based on a given threshold value, setting values to 1 or 0 accordingly."""
    binarized = np.where(image > threshold_value, 1, 0)

    return binarized


@njit
def transform(image, kind, kernel=np.ones((3, 3))):
    """Simple image transformation using one of two available filter functions: Erosion and Dilation.

    Args:
        image:
        kind: Can be either 'erosion', in which case the :func:np.max function is called,
        or 'dilation', when :func:np.min is used instead.
        kernel: n x n kernel with shape < :attr:image.shape, to be used when applying convolution to original image

    Returns:

    """
    if kind == "erosion":
        constant = 1
        apply = np.max
    else:
        constant = 0
        apply = np.min

    center_x, center_y = (x // 2 for x in kernel.shape)

    # Use padded image when applying convolotion to not go out of bounds of the original the image
    transformed = np.zeros(image.shape, dtype=np.uint8)
    padded = np.pad(image, 1, 'constant', constant_values=constant)

    for x in range(center_x, padded.shape[0] - center_x):
        for y in range(center_y, padded.shape[1] - center_y):

            center = padded[x - center_x: x + center_x + 1, y - center_y: y + center_y + 1]
            # Apply transformation method to the centered section of the image
            transformed[x - center_x, y - center_y] = apply(center[kernel == 1])

    return transformed


@njit
def opening_filter(image, kernel=np.ones((3, 3))):
    """Opening filter, defined as the sequence of erosion and then a dilation filter on the same image."""
    return transform(
        transform(image, 'dilation', kernel),
        'erosion', kernel
    )


@njit
def closing_filter(image, kernel=np.ones((3, 3))):
    """Opening filter, defined as the sequence of dilation and then erosion filter on the same image."""
    return transform(
        transform(image, 'erosion', kernel),
        'dilation', kernel
    )


@njit
def binary_mask(image_gray, image_map):
    """Apply binary mask, or thresholding based on bit mask value (mapping mask is 1 or 0)."""
    true_mask, false_mask = np.array(image_gray, copy=True), np.array(image_gray, copy=True)
    true_mask[image_map == 1] = 1
    false_mask[image_map == 0] = 0

    return true_mask, false_mask


@njit
def matrix_concurrency(image, coordinate):
    """Calculate sample co-occurrence matrix based on input image as well as selected coordinates on image.
    Implementation is made using basic iteration, as function to be performed (np.max) is non-linear and therefore
    not usable on the Fourier Transform domain."""
    matrix = np.zeros([np.max(image) + 1, np.max(image) + 1])

    offset_x, offset_y = coordinate[0], coordinate[1]

    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):

            base_pixel = image[x, y]
            offset_pixel = image[x + offset_x, y + offset_y]

            matrix[base_pixel, offset_pixel] += 1

    return matrix / np.sum(matrix)


@njit
def haralick_descriptors(matrix):
    """Calculates all 8 Haralick descriptors based on co-occurence input matrix.
    All descriptors are as follows:
    Maximum probability, Inverse Difference, Homogeneity, Entropy, Energy, Dissimilarity, Contrast and Correlation

    Args:
        matrix: Co-occurence matrix to use as base for calculating descriptors.

    Returns:
        Reverse ordered list of resulting descriptors
    """
    # Function np.indices could be used for bigger input types, but np.ogrid works just fine
    i, j = np.ogrid[0:matrix.shape[0], 0:matrix.shape[1]]  # np.indices()

    # Pre-calculate frequent multiplication and subtraction
    prod = np.multiply(i, j)
    sub = np.subtract(i, j)

    # Calculate numerical value of Maximum Probability
    maximum_prob = np.max(matrix)
    # Using the definition for each descriptor individually to calculate its matrix
    correlation = prod * matrix
    energy = np.power(matrix, 2)
    contrast = matrix * np.power(sub, 2)

    dissimilarity = matrix * np.abs(sub)
    inverse_difference = matrix / (1 + np.abs(sub))
    homogeneity = matrix / (1 + np.power(sub, 2))
    entropy = -(matrix[matrix > 0] * np.log(matrix[matrix > 0]))

    # Sum values for descriptors ranging from the first one to the last, as all are their respective origin matrix
    # and not the resulting value yet.
    descriptors = [maximum_prob, correlation.sum(), energy.sum(), contrast.sum(),
                   dissimilarity.sum(),  inverse_difference.sum(), homogeneity.sum(), entropy.sum()]
    return descriptors


@njit
def get_descriptors(masks, coordinate):
    """Calculate all Haralick descriptors for a sequence of different co-occurrence matrices, given input
    masks and coordinates."""

    descriptors = list()
    for mask in masks:
        descriptors.append(haralick_descriptors(
            matrix_concurrency(mask, coordinate))
        )
    # Concatenate each individual descriptor into one single list containing sequence of descriptors
    return np.concatenate(descriptors, axis=None)


@njit
def euclidean(point_1: np.ndarray, point_2: np.ndarray):
    """Simple method for calculating the euclidean distance between two points, with type np.ndarray."""
    return np.sqrt(np.sum(np.square(point_1 - point_2)))


@njit
def get_distances(descriptors, base):
    """Calculate all Euclidian distances between a selected base descriptor and all other Haralick descriptors
    The resulting comparison is return in decreasing order, showing which descriptor is the most similar to the
    selected base.

    Args:
        descriptors: Haralick descriptors to compare with base index
        base: Haralick descriptor index to use as base when calculating respective euclidean distance to other descriptors.

    Returns:
        Ordered distances between descriptors

    """
    distances = []

    for description in descriptors:
        distances.append(euclidean(description, descriptors[base]))
    # Normalize distances between range [0, 1]
    distances = normalize_array(distances, 1)
    return sorted(enumerate(distances), key=lambda tup: tup[1])


def main():
    # Index to compare haralick descriptors to
    index = int(input())
    q_value = [int(value) for value in input().split()]

    # Format is the respective filter to apply, can be either 1 for the opening filter or else for the closing
    parameters = {'format': int(input()), 'threshold': int(input())}

    # Number of images to perform methods on
    b_number = int(input())

    files, descriptors = (list(), list())

    for _ in range(b_number):
        file = input().rstrip()
        files.append(file)

        # Open given image and calculate morphological filter, respective masks and correspondent Harralick Descriptors.
        image = imageio.imread(file).astype(np.float32)
        gray = grayscale(image)
        threshold = binarize(gray, parameters['threshold'])

        morphological = opening_filter(threshold) if parameters['format'] == 1 else closing_filter(threshold)
        masks = binary_mask(gray, morphological)
        descriptors.append(get_descriptors(masks, q_value))

    # Transform ordered distances array into a sequence of indexes corresponding to original file position
    distances = get_distances(descriptors, index)
    indexed_distances = np.array(distances).astype(np.uint8)[:, 0]

    # Finally, print distances considering the Haralick descriptions from the base file to
    # all other images using the morphology method of choice.
    print(f"Query: {files[index]}")
    print("Ranking:")
    for idx, file_idx in enumerate(indexed_distances):
        print(f"({idx}) {files[file_idx]}", end="\n")


if __name__ == "__main__":
    main()
