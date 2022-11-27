from model import Trainer
from prepossing import data_preprocessing
import numpy as np
import pandas as pd

data = data_preprocessing("E1.wav")
x_train = data.values
trainer = Trainer("models/binary_crossentropy.h5")
trainer.load()
prediction = trainer.model().predict(data)
chords = ["A", "B", "C", "D", "E", "F", "G"]

prediction = prediction.tolist()

j = 0
for i in prediction:
    for k in i:
        if k == max(i):
            print("The sound is {} with {} % accuracy".format(
                chords[j], k))

        else:
            j += 1
