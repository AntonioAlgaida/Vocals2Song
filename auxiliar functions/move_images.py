#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:44:59 2019

@author: Antonio Guillen-Perez
@email: antonio.guillen@edu.upct.es
"""
import os

def move_images():
    '''
    This function can be used to move the images creates by the sound2image_mp.py file to input and output folders.
    '''
    for path, subdir, files in os.walk('musdb18hq/images/'):
        for file in [f for f in files]:
            print(file)
            dst = 'input/' if file.startswith('vocals') else 'output/'
            new_name = '_'.join(file.split('_')[1:])
            os.rename(path + file, 'musdb18hq/images/' + dst + new_name)


if __name__ == "__main__":
    move_images()