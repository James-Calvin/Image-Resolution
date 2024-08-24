"""
Title: Assignment 1: Image Resolution
Author: James-Calvin
Date: 2024-08-22
Description: CS4732 Machine Vision Assignment 1 submission
"""

import numpy as np
from PIL import Image


def down_sample(image, factor = 2):
  """
  Down-sample the image by taking every factor-th pixel
  :param image: The image to down-sample
  :param factor: The factor to down-sample by
  :return: The down-sampled image
  """
  # Convert the image to a numpy array
  image_array = np.array(image)
  
  # Loop over the rows and columns of the image by the factor
  # which results in a smaller array by the factor value
  image_array = image_array[::factor, ::factor]
  
  # Convert the array back into an ImageFile and return
  return Image.fromarray(image_array)

def upsample(image, factor = 2):
  """
  Upsample the image by filling in empty pixels with preceding known value
  :param image: The ImageFile to upsample
  :param factor: The factor to upsample by
  :return: The up-sampled image
  """

  # Convert to array and obtain size
  image_array = np.array(image)
  height, width = image_array.shape

  # Calculate the the size after upscaling
  new_height = height * factor
  new_width = width * factor

  # Instantiate an empty array of the new size
  up_sampled_image_array = np.zeros((new_height, new_width), dtype=np.uint8)

  # Fill each pixel with the preceding known value
  for i in range(new_height):
    for j in range(new_width):
      up_sampled_image_array[i,j] = image_array[i//factor, j//factor]

  # Return in the same format we accepted the image in
  return Image.fromarray(up_sampled_image_array)

def resample(image, factor = 2):
  """
  Performs a down-sample then upsample to maintain original resolution
  :param image: The ImageFile to resample
  :param factor: The factor to resample by
  :return: The resampled image
  """
  down_sampled_image = down_sample(image, factor)
  return upsample(down_sampled_image, factor)

def gray_level_reduction(image, factor = 2):
  """
  Reduces the color resolution from 256 by factor
  :param image: The ImageFile to reduce
  :param factor: The factor to reduce by
  :return: The reduced image
  """
  image_array = np.array(image)

  # steps = 256 // factor
  image_array = (image_array // factor) * factor

  return Image.fromarray(image_array)

if __name__ == "__main__":
  # Load the image
  image_path = 'rose.jpg'
  image = Image.open(image_path)
  
  # Scale the original image each time so that if we implement
  # A different down-sampling method, error will not propagate
  resample(image,1).save("resampled_by_1.jpg") # Sanity check
  resample(image).save("resampled_by_2.jpg") # Our 512x512 image
  resample(image,4).save("resampled_by_4.jpg") # Our 256x256 image
  resample(image,8).save("resampled_by_8.jpg") # Our 128x128 image
  resample(image,16).save("resampled_by_16.jpg") # Our 64x64 image
  resample(image,32).save("resampled_by_32.jpg") # Our 32x32 image
  resample(image,5).save("resampled_by_5.jpg") # Non-power of 2 factor for fun
  
  # Gray level reduction
  gray_level_reduction(image,2).save("grayscale_reduction_by_2.jpg")
  gray_level_reduction(image,4).save("grayscale_reduction_by_4.jpg")
  gray_level_reduction(image,8).save("grayscale_reduction_by_8.jpg")
  ### Below are unnecessary for assignment, but performed for curiosity ###
  gray_level_reduction(image,16).save("grayscale_reduction_by_16.jpg")
  gray_level_reduction(image,32).save("grayscale_reduction_by_32.jpg")
  gray_level_reduction(image,64).save("grayscale_reduction_by_64.jpg")
