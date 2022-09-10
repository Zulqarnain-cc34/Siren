import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from Siren import ImageSiren, AudioDataset, ImplicitAudioWrapper

path = "../data/audio.wav"
audio_dataset = AudioDataset(path)
coord_dataset = ImplicitAudioWrapper(audio_dataset)


# Parameters
n_epochs = 30
logging_freq = 3


model_name = "siren"  # "siren", "mlp_relu"
hidden_features = 256
hidden_layers = 3
lr = 1e-4
logging_root = f"../visualization/{model_name}/audio/"


# Model creation
if model_name == "siren":
    model = ImageSiren(
        hidden_features=hidden_features,
        hidden_layer=hidden_layers,
        hidden_omega=30,
    )

dataloader = DataLoader(coord_dataset, batch_size=1,
                        shuffle=True, pin_memory=True)
optim = torch.optim.Adam(lr=lr, params=model.parameters())

# Training loop
for e in range(n_epochs):
    losses = []
    for batch in dataloader:
        waveform = batch[1]["func"]
        rate = batch[1]["rate"]
        print("____________predicting___________")
        pred = model(waveform)
        print("____________predicted_____________")

        loss = ((waveform-pred)**2).mean()
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(e, np.mean(losses))
    if (e % logging_freq == 0):
        waveform = torch.squeeze(pred).detach().numpy()
        rate = torch.squeeze(batch[1]["rate"]).detach().numpy()
        filename = os.path.join(logging_root, f"pred_waveform_{lr}_{e}.wav")
        write(filename, rate, waveform)
