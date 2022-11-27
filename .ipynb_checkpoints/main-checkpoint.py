import librosa
import numpy as np
import os
from IPython import display
import tensorflow as tf
import IPython.display as ipd
from tensorflow import keras

from tensorflow.keras import layers , models
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
import soundfile as sf

sr = 22050 # sample rate
T = 5.0    # seconds
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
x = 2.5*np.sin(2*np.pi*250*t)# pure sine wave at 220 Hz
#Playing the audio
ipd.Audio(x, rate=sr) # load a NumPy array
sf.write("new_audio.wav", x , sr)
# path = os.getcwd()
# audio_data = 'cat.wav'

# x , sr = librosa.load(audio_data)
# plt.figure(figsize = (14,5))
# librosa.display.waveplot(x , sr = sr)