import pickle
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np
import librosa
from natsort import natsorted
from matplotlib import pyplot
from tqdm import tqdm
location = '/home/pedrolouro/experiments-msc/New_Public_DB_Data/total_wav/'

fftsize = 1024
window = 1024
hop = 512
melBin = 128


def get_file_names():
    curr_path = Path('../gen_csv_with_targets.py').absolute().parent.parent.parent.parent.__str__()
    ptd = '/New_Public_DB_Data/new_pub_complete/'

    files_q1 = [curr_path + ptd + 'Q1/' + f for f in listdir(curr_path + ptd + 'Q1/')
                if isfile(join(curr_path + ptd + 'Q1/', f))]
    files_q2 = [curr_path + ptd + 'Q2/' + f for f in listdir(curr_path + ptd + 'Q2/')
                if isfile(join(curr_path + ptd + 'Q2/', f))]
    files_q3 = [curr_path + ptd + 'Q3/' + f for f in listdir(curr_path + ptd + 'Q3/')
                if isfile(join(curr_path + ptd + 'Q3/', f))]
    files_q4 = [curr_path + ptd + 'Q4/' + f for f in listdir(curr_path + ptd + 'Q4/')
                if isfile(join(curr_path + ptd + 'Q4/', f))]

    all_files = []
    for files in [files_q1, files_q2, files_q3, files_q4]:
        all_files.extend(files)

    all_files = natsorted(all_files, key=lambda all_files: all_files.split('/')[-1])
    return all_files

###########################################
# Code from: ################################
# https://github.com/jongpillee/ismir2018-artist~ ###
###########################################
def get_mels():
    files = get_file_names()

    all_mels = []
    for fi in tqdm(files):
        y, sr = librosa.load(fi, sr=22050)
        S = librosa.core.stft(y, n_fft=fftsize, hop_length=hop, win_length=window)
        X = np.abs(S)

        mel_basis = librosa.filters.mel(sr, n_fft=fftsize, n_mels=melBin)

        mel_S = np.dot(mel_basis, X)

        mel_S = np.log10(1 + 10 * mel_S)
        mel_S = mel_S.astype(np.float32)

        mel_S = mel_S[:, :1291]
        mel_S = np.rot90(np.pad(mel_S, ((0, 0), (0, 1291 - mel_S.shape[1]))))
        all_mels.append(mel_S)

    with open('dataset_22_05kHz_norm_lee.pickle', 'wb') as a_file:
        pickle.dump(all_mels, a_file)

if __name__ == '__main__':
    get_mels()

