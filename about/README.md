# Segmentation using basic filters and algorithms

## Otsu's Thresholding Filter

After having performed the basic preprocessing tasks for the dataset images, we decided to use a color-based (RGB) thresholding filter, to try and threshold only the colored pixels that match the top view of houses and buildings. Since most buildings were similarly colored when in close proximity, we thought this would be a good initial guess as to try and generate a fitting mask.

## Watershed Gradient

The Otsu filter didn't quite match expectations, as even though it did correctly segment most buildins and houses, it also included large bodies of water, as well as large areas that had matching colors or color patterns. Therefore, we decided to use a segmentation algorithm based on regions. We ended up using the Watershed algorithm, as it performed better than the other algorithms used during the Image Processing classes. We ended up only using the Gradient matrix used during the Watershed algorithm, as it better separated the buildings from the streets and lakes as the final output of the Watershed algorithm.

## Basic segmentation generated images

Below, we generated some Matplotlib figures containing the original image, the Otsu Threshold mask, the Watershed Gradient mask, as well as the true mask from within the dataset. For the entire dataset, the Otsu Treshold obtained the lowest mean MSE, of about `30811`.

[Generated Images](https://github.com/rzimmerdev/pdi-2022/tree/main/predictions/filters):

<img src="https://github.com/rzimmerdev/pdi-2022/blob/1b7d6370b6d4d9163a56ea0f41f2216d3a3aecc2/predictions/filters/img_1.png" width="900" height="400" />
<img src="https://github.com/rzimmerdev/pdi-2022/blob/1b7d6370b6d4d9163a56ea0f41f2216d3a3aecc2/predictions/filters/img_6.png" width="900" height="400" />
<img src="https://github.com/rzimmerdev/pdi-2022/blob/1b7d6370b6d4d9163a56ea0f41f2216d3a3aecc2/predictions/filters/img_9.png" width="900" height="400" />


# Segmentation using a Convolutional Neural Network

For this specific problem, we ended up using the UNet, as it performs better with smaller datasets, such as the one we had. Since the two previous methods had some very specific problems, intrinsically related to the idea behind the algorithms (such as being color or region dependant), using a neural network seemed to be more adequate for this task.

The UNet implementation can be found [here](https://github.com/rzimmerdev/pdi-2022/blob/main/scripts/unet.py).

The Torch library was used to implement and train the network (which was trained using a GPU and requires one to retrain for its current implementation).
Overall, the model performed much better than either the Otsu filter or the Watershed segmentation, as the following images also show:

[Predicted Masks](https://github.com/rzimmerdev/pdi-2022/tree/main/predictions/unet)


<img src="https://github.com/rzimmerdev/pdi-2022/blob/1b7d6370b6d4d9163a56ea0f41f2216d3a3aecc2/predictions/unet/img_0.png" width="900" height="400" />
<img src="https://github.com/rzimmerdev/pdi-2022/blob/1b7d6370b6d4d9163a56ea0f41f2216d3a3aecc2/predictions/unet/img_1.png" width="900" height="400" />
<img src="https://github.com/rzimmerdev/pdi-2022/blob/1b7d6370b6d4d9163a56ea0f41f2216d3a3aecc2/predictions/unet/img_8.png" width="900" height="400" />
