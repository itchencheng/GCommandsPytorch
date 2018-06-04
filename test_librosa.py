
import librosa
import numpy as np
import torch

import sys

import matplotlib.pyplot as plt

def main(path):
    window="hamming"
    window_size = 0.02
    window_stride=0.01
    max_len=101
    normalize=True


    # sr means sampe-rate
    y, sr = librosa.load(path, sr=None)
    print(('y',(len(y), np.max(y), np.min(y))))
    print(('sr',sr))

    if(0):
        plt.plot(y)
        plt.show()

    n_fft = int(sr * window_size)
    print(('n_fft', n_fft))
    win_length = n_fft
    hop_length = int(sr * window_stride)
    print(('win_length', win_length))
    print(('hop_length', hop_length))
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    print(('D.shape', D.shape))
    print(('spect.shape', spect.shape))

    if(1):
        plt.plot(D[:,0])
        plt.show()

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]

    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    print(spect.shape)
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


if __name__ == "__main__":
    wav_file = 'speech_commands/train/bed/0a7c2a8d_nohash_0.wav'
    main(wav_file)
