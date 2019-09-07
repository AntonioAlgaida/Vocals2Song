# Vocals2Song
Tensorflow implementation of pix2pix for creating music from a voice signal.

## Description:
This project consists in the creation of music from a voice audio, using the implementation of Tensorflow pix2pix. It is approached as an inverse problem of separation of components in a song.  I have pre-processed the raw data (vocals and mixture pair dataset) in an image that contains encoded information provided by the spectrogram of the signals, which can be treated as a 2-D image to train the pix2pix model. 

## Requirements:
   * NumPy >= 1.11.1
   * TensorFlow >= 1.0.0
   * librosa

## Dataset:
The dataset used is the musdb18hq, which is widely used in the field of Deep learning for tasks of Music Unmixing, ie for a song (mixture), is intended to isolate each of the components that make up a song. 
The musdb18 is a dataset of 150 full lengths music tracks (~10h duration) of different genres along with their isolated drums, bass, vocals and others stems.
musdb18 contains two folders, a folder with a training set: "train", composed of 100 songs, and a folder with a test set: "test", composed of 50 songs. Supervised approaches should be trained on the training set and tested on both sets.
All signals are stereophonic and encoded at 44.1kHz.
As an alternative, we also offer the uncompressed WAV files for models that aim to predict high bandwidth of up to 22 kHz. Other than that, MUSDB18-HQ is identical to MUSDB18.

## Data pre-processing:
For the data pre-processing, first, the spectrogram of an audio signal of 15 seconds is obtained (for that of standardizing sizes, although they could be filled with zeros if the signal has a duration of that one) and this spectrogram is encoded in an image. In this coding, in the component R (Red) of the image is coded the module of the spectrogram, that is to say, the power of each one of the frequencies in all the temporal range and, in the component G (Green) of the image is coded the phase of the signal, something very important if we want to reconstruct the original signal, since not only we need the module, but also the phase. Finally, component B (Blue) is imposed to be 0. That image is saved with a data format that is able to read Tensorflow as it is .bmp. For it, each one of the components is quantified between integer values from 0 to 255, since .bmp is an image format of 8 bits for each component.
As the signal we are processing is sampled at 44.1kHz, with a duration of 15 seconds per signal, and the spectrogram is obtained with a fft of size 1024 and with a jump between ffts of 256 samples, the result of processing each of the signals gives us an image of size 2584 x 513 pixels. Remembering that in each one of these pixels is coded the module and the phase for each one of the frequencies in all the temporal interval, whose ranges are quantized between 0 and 255, being images .bmp.
## Results:
TBC!!

## Where can this approach be applied?
   * Applications where you enter a voice and get a music appropriate to that voice.
   * Create music cover, where the network only trains with voice and (voice+piano) pairs, or something like this.
   * To change the style of a song, starting from a song of one and starting from another. For example, go from a song to a rock song and turn it into traditional folklore (with a secure dataset obtained from digital audio platforms).
   * Music unmixing. From mixture to each of the components that make up a song.
   * Identification and separation of voices in noisy environments or with many voices.
   * Voice changes male, female.
   
Basically any task where it involves audio signals and there is an adequate dataset.

## Main problems encountered and possible improvements:
   * Due to the quantification of the sound spectrogram to an 8-bit coded .bpm image (integer values between 0 and 255), quite a huge quantification noise is generated (see https://en.wikipedia.org/wiki/Quantization_(signal_processing)#Quantization_noise_model).
   * Other formats could be used such as .tiff, which allows images to be encoded in 16 bits per channel, but Tensorflow does not allow .tiff images to be read at present (07/09/2019).
   * It could be encoded in an image that has 4 components (such as RBGA), to reduce quantization noise. This could be done by chopping the module of the spectrogram into 2 ranges ([min, median), [median, max]), and encoding these 2 ranges separately, reducing the encoding noise. The same could be done with phase, instead of encoding it into one component, you can encode into 2 components, splitting the range into 2, and encoding each into a different component.
   
## To go further:
The Fourier transform (FT) decomposes a function of time (a signal) into its constituent frequencies. This is similar to the way a musical chord can be expressed in terms of the volumes and frequencies of its constituent notes. The term Fourier transform refers to both the frequency domain representation and the mathematical operation that associates the frequency domain representation to a function of time. The Fourier transform of a function of time is itself a complex-valued function of frequency, whose magnitude (modulus) represents the amount of that frequency present in the original function, and whose argument is the phase offset of the basic sinusoid in that frequency. (See: https://en.wikipedia.org/wiki/Fourier_transform)

The spectrogram is the result of calculating the spectrum of a signal. It is a three-dimensional graph that represents the energy of the frequency content of the signal as it varies over time. It is used, for example, to identify phonetic sounds and speech processing, for radar/sonar. The spectrogram is a basic representation tool that is used for the analysis of electrical signals, communications, and any audiovisual signal in its frequency content. The representation of the spectrum of a signal in the frequency domain can help to better understand its content, than with a representation in the time domain.

![Spectrogram of violin](https://github.com/AntonioAlgaida/Vocals2Song/blob/master/Spectrogram_of_violin.png)
A spectrogram of violin sound. The bright lines at the bottom are the fundamentals of each note and the other bright lines nearby are the harmonics; together, they form the frequency spectrum.

To calculate the spectogram, the `librosa.stft` function has been used, which returns the STFT matrix (see https://en.wikipedia.org/wiki/Short-time_Fourier_transform), where, in each of the cells of the matrix contain a complex number with real part and imaginary part.
To calculate the audio signal from the coded image, the `librosa.istft` function has been used, where the input parameters is the matrix previously commented, and the parameters used for the fourier transforms.

This project is part of the competition organized by the youtube channel [dotcsv](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg), which I strongly recommend to subscribe. 
\#RetoDotCSV2080Super