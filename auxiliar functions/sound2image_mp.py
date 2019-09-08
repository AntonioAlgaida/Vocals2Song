#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:36:18 2019

@author: Antonio Guillen-Perez
@email: antonio_algaida@hotmail.com

"""
import os
import multiprocessing
import imageio
import librosa

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile
from multiprocessing import Pool


def create_chunks():
    '''
    Obtains the relative paths of all sounds (voices and mixtures).
    '''
    paths = []
    data_sources = ['Sources', 'Mixtures']
    train_tests = ['train', 'test']
    for data_source in data_sources:
        for train_test in train_tests:
            for path, subdirs, files in os.walk('musdb18hq/{}'.format(train_test)):
                    if data_source == 'Mixtures':
                        end = 'mixture.wav'
                    else:
                        end = 'vocals.wav'

                    for name in [f for f in files if f.endswith(end)]:
                        paths.append(os.path.join(path, name))
    return paths


def obtain_spectrogram_mp(path):
    # Params to tweek depends on your dataset.
    fft_size = 1024
    hop_length = int(fft_size/4)
    window_size = fft_size
    fs, samples = wavfile.read(path)

    # From stereo to mono.
    samples = samples.sum(axis=1) / 2

    # The duration is 15 seconds.
    duration = fs*15 # 15 seconds

    # I skip the first samples, they are all zeros.
    skip = 50000

    # Auxiliar variables.
    j = 0
    for i in range(skip, len(samples)-skip, duration):
        # Obtain the subsample of a sound
        samples2 = samples[i:i+duration]
        if samples2.shape[0] == duration:
            out_path = path.split('/')[:-1]
            train_test = path.split('/')[-1].replace('.wav','')
            out_path = '/'.join(out_path)
            img_path = path.replace(' ', '').replace("'", '').replace('-','').replace('.wav','')
#            out_path = out_path.replace(' ', '').replace("'", '').replace('-','').replace('.wav','')
            print(j)
            if not os.path.isfile('{}_{}.bmp'.format(img_path,j)):
                # Spectrogram
#                f, t, Zxx = signal.stft(samples2, fs)
                Zxx = librosa.stft((samples2).astype(np.float64),
                                   n_fft=fft_size,
                                   hop_length=hop_length,
                                   win_length=window_size)

                # Red component: R. The module of the spectrum.
                R = np.abs(Zxx).astype(np.float64) # range (0, 16383)
                R -= R.min()

                # Green component: G. The phase of the spectrum.
                G = np.angle(Zxx).astype(np.float64) # range [-pi:pi]
                G += np.pi # range [0:2*pi]

                # Blue component: B. Always zero.
                # In a future, It can be used in order to reduce the sampling noise in
                # the module of the spectrum, because the samples are confined in the
                # low power values
                B = np.zeros_like(R)

                comp_max = [16384, 2*np.pi]

                thr = np.quantile(R, 0.99)
                R[R>=thr] = thr

                # Normalize the range between 0 and 255, because the bmp format only admits
                # 8 bits for channel.
                R_norm = (R * (255 / comp_max[0])).astype(np.uint8)
                G_norm = (G * (255 / comp_max[1])).astype(np.uint8)
                B_norm = np.zeros_like(R_norm)

                # Create the codificated spectrum.
                img = np.stack((R_norm, G_norm, B_norm), axis = 2)

                # Save the codificated spectrum
#                imageio.imwrite('_{}.tiff'.format(j), img)
                out_path2 = '/'.join(path.split('/')[:-2])
                sound = img_path.split('/')[2]
                imageio.imwrite('{}/{}_{}_{}.bmp'.format(out_path2,train_test,sound,j), img)

            else:
                print('{}_{}.bmp already exist'.format(path,j))
            j += 1
#%%
if __name__ == "__main__":
    N_CPUS = multiprocessing.cpu_count()

    paths = create_chunks()

    number_of_workers = N_CPUS-1
    with Pool(number_of_workers) as p:
        p.map(obtain_spectrogram_mp, paths)


