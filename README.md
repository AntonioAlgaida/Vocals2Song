# Vocals2Song
Tensorflow implementation of pix2pix for creating music from a voice signal.


## Description:
This project consists in the creation of music from a voice audio, using the implementation of Tensorflow pix2pix. It is approached as an inverse problem of separation of components in a song. The main idea is to use this network to create music, only by introducing voice. To do this, the network will be trained with a dataset with pairs of voices / mixures (voices + piano + guitar + bass + drums + ...). I have pre-processed the raw data (vocals and mixture pair dataset) in an image that contains encoded information provided by the spectrogram of the signals, which can be treated as a 2-D image to train the pix2pix model. 


## Pix2Pix model:
Pix2Pix is a Generative Adversarial Network, or GAN, model designed for general purpose image-to-image translation, trained on paired examples.

For example, the model can be used to translate images of daytime to nighttime, or from sketches of products like shoes to photographs of products.

The GAN architecture is comprised of a generator model for outputting new plausible synthetic images and a discriminator model that classifies images as real (from the dataset) or fake (generated). The discriminator model is updated directly, whereas the generator model is updated via the discriminator model. As such, the two models are trained simultaneously in an adversarial process where the generator seeks to better fool the discriminator and the discriminator seeks to better identify the counterfeit images.

The Pix2Pix model is a type of conditional GAN, or cGAN, where the generation of the output image is conditional on an input, in this case, a source image. The discriminator is provided both with a source image and the target image and must determine whether the target is a plausible transformation of the source image.

The benefit of the Pix2Pix model is that compared to other GANs for conditional image generation, it is relatively simple and capable of generating large high-quality images across a variety of image translation tasks.

Generative Adversarial Network framework:
![Gan architecture](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/GANs.png)
![Image credits](https://www.freecodecamp.org/news/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394): https://www.freecodecamp.org/news/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394/


## Requirements:
   * NumPy >= 1.11.1
   * TensorFlow >= 1.0.0
   * librosa


## Base Dataset:
The dataset used is the musdb18hq, which is widely used in the field of Deep learning for tasks of Music Unmixing, ie for a song (mixture), is intended to isolate each of the components that make up a song. 
The musdb18 is a dataset of 150 full lengths music tracks (~10h duration) of different genres along with their isolated drums, bass, vocals and others stems.
musdb18 contains two folders, a folder with a training set: "train", composed of 100 songs, and a folder with a test set: "test", composed of 50 songs. Supervised approaches should be trained on the training set and tested on both sets.
All signals are stereophonic and encoded at 44.1kHz.
As an alternative, we also offer the uncompressed WAV files for models that aim to predict high bandwidth of up to 22 kHz. Other than that, MUSDB18-HQ is identical to MUSDB18.

![Source Dataset](https://sigsep.github.io/datasets/musdb.html)https://sigsep.github.io/datasets/musdb.html


## Data pre-processing:
For the data pre-processing, first, the spectrogram of an audio signal of 15 seconds is obtained (for that of standardizing sizes, although they could be filled with zeros if the signal has a duration of that one) and this spectrogram is encoded in an image. In this coding, in the component R (Red) of the image is coded the module of the spectrogram, that is to say, the power of each one of the frequencies in all the temporal range and, in the component G (Green) of the image is coded the phase of the signal, something very important if we want to reconstruct the original signal, since not only we need the module, but also the phase. Finally, component B (Blue) is imposed to be 0. That image is saved with a data format that is able to read Tensorflow as it is .bmp. For it, each one of the components is quantified between integer values from 0 to 255, since .bmp is an image format of 8 bits for each component.
As the signal we are processing is sampled at 44.1kHz, with a duration of 15 seconds per signal, and the spectrogram is obtained with a fft of size 1024 and with a jump between ffts of 256 samples, the result of processing each of the signals gives us an image of size 2584 x 513 pixels. Remembering that in each one of these pixels is coded the module and the phase for each one of the frequencies in all the temporal interval, whose ranges are quantized between 0 and 255, being images .bmp.

After this processing we obtain the following:

Encoded voice: ![Encoded voice](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/examples/1_input.png)

Encoded mixture: ![Encoded mixture](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/examples/1_output.png)


## Dataset created:
The dataset created are 2274 images of voices (~16GB) and it's 2274 images of mixtures (~16GB).
Each image has a resolution of 2056 x 513 pixels.
This resolution is important, as each pixel of the image contains information both temporally and frequently. Therefore, it is not possible to change the resolution of the image to a smaller size. This damages the performance of the network, but with adequate hardware, the performance is adequate. 


## Where can this approach be applied?
   * Applications where you enter a voice and get a music appropriate to that voice.
   * Create music cover, where the network only trains with voice and (voice+piano) pairs, or something like this.
   * To change the style of a song, starting from a song of one and starting from another. For example, go from a song to a rock song and turn it into traditional folklore (with a secure dataset obtained from digital audio platforms).
   * Music unmixing. From mixture to each of the components that make up a song.
   * Identification and separation of voices in noisy environments or with many voices.
   * Voice changes male, female.
   
Basically any task where it involves audio signals and there is an adequate dataset.


## Results and Conclusions:
|           | Audio | Spectrogram |
| --------- | ------------- | ------------- |
| Voice     | ![Audio](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Results/input.wav)   | ![](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Results/input.png)  |
| Mixture    | ![Audio](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Results/output.wav)   | ![](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Results/output.png)  |
| Mixture predicted | ![Audio](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Results/predicted.wav)  | ![](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Results/predicted.png)  |

Training GIF:
![Training Gif](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Results/training_evolution.gif)

The results, while not yet showing impressive results, show that this type of GAN networks could be used for audio processing, with a previous transformation of the spectrogram to a suitable image.

Other results can be heard on:
![Other Results](https://github.com/AntonioAlgaida/Vocals2Song/tree/master/Results/Others)

Where X_input{.wav/.bmp} is the audio containing only the vocals, X_output{.wav/.bmp} is the audio containing the mixture audio and X_predicted{.wav/.bmp} is the mixture audio predicted by the Pix2Pix network performed in this work. X is an integer value between 0 and 7, which indicates each of the test audios.

The Pix2Pix network has been trained for 24 hours in Colab de google. 

The final results obtained on 14/09/2019 at 23:19 are shown below.
![Last results on 14/09/2019:](https://github.com/AntonioAlgaida/Vocals2Song/tree/master/Results)
These results are {130,170,178,189,196}_notrain_0.wav_

Although the final results obtained are not impressive, they show promising results.

## Main problems encountered and possible improvements:
   * Due to the quantification of the sound spectrogram to an 8-bit coded .bpm image (integer values between 0 and 255), quite a huge quantification noise is generated (see https://en.wikipedia.org/wiki/Quantization_(signal_processing)#Quantization_noise_model).
   * Other formats could be used such as .tiff, which allows images to be encoded in 16 bits per channel, but Tensorflow does not allow .tiff images to be read at present (07/09/2019).
   * It could be encoded in an image that has 4 components (such as RBGA), to reduce quantization noise. This could be done by chopping the module of the spectrogram into 2 ranges ([min, median), [median, max]), and encoding these 2 ranges separately, reducing the encoding noise. The same could be done with phase, instead of encoding it into one component, you can encode into 2 components, splitting the range into 2, and encoding each into a different component.
   * Just predict the mix of instruments (piano, guitar, bass, violin, etc) and join with the vocal input signal.
   * Due to problems with RAM on google colab, I can only train the model with 42 images and test with 8. I think that by reducing the images size the number of images for training can be increased.
   * Due to the previous, the subset of the dataset used don't allow that the pix2pix network doesn't learn a lot. With more ram I expect more interesting results. 
   * Perhaps, the use of a dataset containing electronic music, which contains musical notes more marked and contained in time, could improve the performance of the network.
   * Due to the low number of training images (the network was trained with only 42 images) the network was trained with 42 images * 15 secs/image = 630 secs = 10.5 mins. This value is very low. It should be corrected in future versions of this study.
   
## How to use the proyect:
In order to use the designed project you must follow the following steps:

1º: Obtain a dataset of pairs of audios to work with.

2º: Pre-process the audio files with the sound2image_mp.py file. It is necessary to adapt the parameters of this file to fit the selected audio files, such as the duration of the treated audio portion, the size of the fourier transform window, etc.

3º: We already have the pairs of images that the pix2pix model needs (voices2mixture.ipyb file), now to train the model. To do this, the model needs in a folder inputs all the images corresponding to the isolated voice signals. In addition it needs in another folder the images of the mixures signals. The model performs the split to obtain the training and test dataset.

4º: Once the model has been trained, in order to hear the result, we have to perform the inverse step to 2, that is, convert the image to sound. To do this, using the file image2sound.py and providing the path to the image we want to convert, we convert it to an audio signal in .wav format.


## Final Conclusions:
Despite the numerous problems detailed in the previous section, the results show are very promising, as long as the problems shown above are solved and the quantification of the image components is improved.
This novel approach can provide numerous advances in multiple fields such as the problem of separating the components of a song, the creation of automatic music from a voice, recreational use as in karaokes, etc.
With a bigger dataset (RAM problem) the use of 16-bits by components images (like .tiff) and a faster machine I expected even better results. 


## To go further:
The Fourier transform (FT) decomposes a function of time (a signal) into its constituent frequencies. This is similar to the way a musical chord can be expressed in terms of the volumes and frequencies of its constituent notes. The term Fourier transform refers to both the frequency domain representation and the mathematical operation that associates the frequency domain representation to a function of time. The Fourier transform of a function of time is itself a complex-valued function of frequency, whose magnitude (modulus) represents the amount of that frequency present in the original function, and whose argument is the phase offset of the basic sinusoid in that frequency. (See: https://en.wikipedia.org/wiki/Fourier_transform)

The spectrogram is the result of calculating the spectrum of a signal. It is a three-dimensional graph that represents the energy of the frequency content of the signal as it varies over time. It is used, for example, to identify phonetic sounds and speech processing, for radar/sonar. The spectrogram is a basic representation tool that is used for the analysis of electrical signals, communications, and any audiovisual signal in its frequency content. The representation of the spectrum of a signal in the frequency domain can help to better understand its content, than with a representation in the time domain.

![Spectrogram of violin](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Spectrogram_of_violin.png)
A spectrogram of violin sound. The bright lines at the bottom are the fundamentals of each note and the other bright lines nearby are the harmonics; together, they form the frequency spectrum.

To calculate the spectogram, the `librosa.stft` function has been used, which returns the STFT matrix (see https://en.wikipedia.org/wiki/Short-time_Fourier_transform), where, in each of the cells of the matrix contain a complex number with real part and imaginary part.
To calculate the audio signal from the coded image, the `librosa.istft` function has been used, where the input parameters is the matrix previously commented, and the parameters used for the fourier transforms.

Example of sound-image-sound process:
   * Origin audio: ![Origin audio](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/origin%20audio.wav)
   * Encoded image: ![Encoded image](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/enconded%20audio.bmp)
   * Decoded audio: ![Decoded audio](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/decoded%20audio.wav)

This project is part of the competition organized by the youtube channel [dotcsv](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg), which I strongly recommend to subscribe if you want to learn machine learning, deep learning, etc, or if you don't want learn about this, you can also subscribe.

## Author:
Antonio Guillen-Perez
antonio_algaida@hotmail.com
https://twitter.com/agnprz

## Dataset Credits:
dataset = musdb18-hq,
Authors = {Rafii, Zafar and Liutkus, Antoine and Stöter, Fabian-Robert and Mimilakis, Stylianos Ioannis and Bittner, Rachel},
title = {MUSDB18-HQ - an uncompressed version of MUSDB18},
month = aug,
year = 2019,
doi = {10.5281/zenodo.3338373},
url = {https://doi.org/10.5281/zenodo.3338373}

