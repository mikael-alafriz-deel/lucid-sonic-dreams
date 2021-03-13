import numpy as np

import PIL
import skimage.exposure


def contrast_effect(array, strength: float, amplitude: float):
  '''Sample effect: increase image contrast'''

  contrast_factor = 1 + amplitude*strength
  img = PIL.Image.fromarray(array)
  contrasted_image = PIL.ImageEnhance.Contrast(img).enhance(contrast_factor)
  return np.array(contrasted_image)


def flash_effect(array, strength: float, amplitude: float):
  '''Sample effect: increase image intensity'''
  
  intensity_factor = 255 - (amplitude*strength*255)
  return skimage.exposure.rescale_intensity(array, (0, intensity_factor))