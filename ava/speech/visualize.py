import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

scale_file = 'audio.wav'
scale, sr = librosa.load(scale_file)
FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print(S_scale.shape)
print(type(S_scale[0][0]))

Y_scale = np.abs(S_scale) ** 2
print(Y_scale.shape)
print(type(Y_scale[0][0]))

plt.figure(figsize=(25, 10))
librosa.display.specshow(Y_scale, sr=sr, hop_length=HOP_SIZE, x_axis= 'time', y_axis='linear')
plt.colorbar(format='%+2.f')

y, sr = librosa.load(scale_file, sr=None)

# generating the Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# converting to log scale (dB)
S_dB = librosa.power_to_db(S, ref=np.max)

# ploting the Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F

S_dB_tensor = torch.tensor(S_dB).unsqueeze(0)

class Conv1DModel(nn.Module):
  def __init__(self):
    super(Conv1DModel, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
    self.gelu = nn.GELU()

  def forward(self, x):
    x = self.conv1(x)
    x = self.gelu(x)
    x = self.conv2(x)
    x = self.gelu(x)
    return x

model = Conv1DModel()
output = model(S_dB_tensor)
print(f"Output shape: {output.shape}")
print(output)