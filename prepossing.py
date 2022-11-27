from os import listdir
from os.path import isfile, join
import librosa
import pandas as pd
import numpy as np
from model import Trainer


def data_preprocessing(file_path):
    column = [[] for i in range(12)]

    data, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(data, sr=sr)

    for i in range(12):
        column[i].append(np.mean(chroma[i]))

    Data = {}
    Chords = ['C', 'C♯', "D", 'D♯', 'E', 'F', "F♯", 'G', 'G♯', 'A', "A♯", 'B']
    for i in range(12):
        Data[Chords[i]] = column[i]

    df = pd.DataFrame(Data)

    return df
