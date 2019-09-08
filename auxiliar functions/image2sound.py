#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:32:11 2019

@author: Antonio Guillen-Perez
@email: antonio_algaida@hotmail.com

"""

import numpy as np
import pandas as pd
import imageio
import librosa
import cv2
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


def image_to_sound(input_image):
    # Params to tweek depends on your dataset.
    fft_size = 1024*1
    hop_length = int(fft_size/4)
    window_size = fft_size
    fs = 44100

    # Load the image
    im_loaded = imageio.imread(input_image)

    # Resize, depend on the pix2pix model
    im_loaded = cv2.resize(im_loaded, dsize=(2584, 513), interpolation=cv2.INTER_CUBIC)

    # Limits of the image to denorm.
    comp_max = [16384*4, 2*np.pi]

    # Module normalizated (mod_norm) and angle normalizated (ang_norm)
    mod_norm = (im_loaded[:,:,0]).astype(np.float64)
    ang_norm = (im_loaded[:,:,1]).astype(np.float64)

    # Denormalization process
    mod = mod_norm * (comp_max[0] / 255)
    ang = ang_norm * (comp_max[1]/ 255)
    ang_comp = ang - np.pi #

    # Makes the complex matrix of the spectrum.
    Zxx_rec = mod+ang_comp*1j

    # Convert from spectrum to audio signal,
    # 661500 is the lenght of the original 15 seg sound
    xrec = librosa.istft(Zxx_rec, hop_length=hop_length, win_length=window_size, length=661500)

    # Save the file.
    wavfile.write('1_input_decoded.wav', fs, (xrec).astype(np.int16))

if __name__ == "__main__":
    input_image = 'input.bmp'
    image_to_sound(input_image)


