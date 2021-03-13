import numpy as np
import random
import pickle 
import requests
import json
import pandas as pd

import librosa
import pygit2
import gdown 
from mega import Mega


def download_weights(url, output):
  '''Download model weights from URL'''

  if 'drive.google.com' in url: 
    gdown.download(url, output = output, quiet = False)

  elif 'mega.nz' in url:
    m = Mega()
    m.login().download_url(url, 
                           dest_filename = output)

  elif 'yadi.sk' in url:
    endpoint = 'https://cloud-api.yandex.net/v1/disk/'\
               'public/resources/download?public_key='
    r_pre = requests.get(endpoint + url)
    r_pre_href = r_pre.json().get('href')
    r = requests.get(r_pre_href)
    with open(output, 'wb') as f:
      f.write(r.content)
      
  else:
    r = requests.get(url)
    with open(output, 'wb') as f:
      f.write(r.content)


def consolidate_models():
  '''Consolidate JSON dictionaries of pre-trained StyleGAN(2) weights'''

  # Define URL's for pre-trained StyleGAN and StyleGAN2 weights
  stylegan_url = 'https://raw.githubusercontent.com/justinpinkney/'\
  'awesome-pretrained-stylegan/master/models.csv'
  stylegan2_url = 'https://raw.githubusercontent.com/justinpinkney/'\
  'awesome-pretrained-stylegan2/master/models.json'

  # Load JSON dictionary of StyleGAN weights
  models_stylegan = pd.read_csv(stylegan_url)\
                    .to_dict(orient='records')

  # Load JSON dictionary of StyleGAN2 weights
  r = requests.get(stylegan2_url)
  models_stylegan2 = json.loads(r.text)

  # Consolidate StyleGAN and StyleGAN2 weights
  all_models = models_stylegan + models_stylegan2

  return all_models


def get_spec_norm(wav, sr, n_mels, hop_length):
  '''Obtain maximum value for each time-frame in Mel Spectrogram, 
     and normalize between 0 and 1'''

  # Generate Mel Spectrogram
  spec_raw= librosa.feature.melspectrogram(y=wav, sr=sr,
                                           n_mels=n_mels,
                                           hop_length=hop_length)
  
  # Obtain maximum value per time-frame
  spec_max = np.amax(spec_raw,axis=0)

  # Normalize all values between 0 and 1
  spec_norm = (spec_max - np.min(spec_max))/np.ptp(spec_max)

  return spec_norm


def interpolate(array_1: np.ndarray, array_2: np.ndarray, steps: int):
  '''Linear interpolation between 2 arrays'''

  # Obtain evenly-spaced ratios between 0 and 1
  linspace = np.linspace(0, 1, steps)

  # Generate arrays for interpolation using ratios
  arrays = [(1-l)*array_1 + (l)*array_2 for l in linspace]

  return np.asarray(arrays)


def full_frame_interpolation(frame_init, steps, len_output):
  '''Given a list of arrays (frame_init), produce linear interpolations between
     each pair of arrays. '''

  # Generate list of lists, where each inner list is a linear interpolation
  # sequence between two arrays.
  frames = [interpolate(frame_init[i], frame_init[i+1], steps) \
            for i in range(len(frame_init)-1)]

  # Flatten list of lists
  frames = [vec for interp in frames for vec in interp]

  # Repeat final vector until output is of the desired length
  while len(frames) < len_output:
    frames.append(frames[-1])

  return frames