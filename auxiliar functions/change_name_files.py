#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 08:57:46 2019

@author: Antonio Guillen-Perez
@email: antonio.guillen@edu.upct.es
"""

import os

def change_name_files():
    '''
    This function changes the names of the images created by the sound2image_mp.py process (and then moved by move_images.py)
    '''
    n = 0
    for path, subdir, files in os.walk('musdb18hq/images/input'):
        for file in [f for f in files]:
            if not file == 'input.rar':
                path_input = os.path.join(path, file)
                new_path_input= os.path.join(path, f'{n:05}.bmp')

                path_output = os.path.join('musdb18hq/images/output', file)
                new_path_output = os.path.join('musdb18hq/images/output', f'{n:05}.bmp')

                os.rename(path_output, new_path_output)
                os.rename(path_input, new_path_input)
                n += 1

if __name__ == "__main__":
    change_name_files()