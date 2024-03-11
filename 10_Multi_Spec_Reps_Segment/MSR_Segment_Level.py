import sys
sys.path.append('../Utils/')
import os
import pickle
import time
from os import listdir, makedirs
from os.path import isfile, join
import random
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
# This call for disabling eager execution is necessary to
# have the ability to wipe models not in use. Attempts to
# wipe a model with eager execution enabled were not successful.
tf.compat.v1.disable_eager_execution()
from keras import Model
from keras.layers import Input, Dense, Flatten, Dropout, ReLU, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.metrics import RootMeanSquaredError
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.backend import clear_session
from pydub import AudioSegment
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import gc
from natsort import natsorted
from tqdm import tqdm
from tensorflow.python.client import device_lib

warnings.simplefilter("ignore")

from MyCustomEarlyStopping import MyCustomEarlyStopping
from History_AV import History

####################
### Options ########
####################

curr_path = Path('.').absolute().parent.__str__()
# Path to the directory with all necessary data
ptd = 'New_MERGE_Complete_Data/'
# Location of the .wav files converted from .mp3 as the intermediate step to generate Mel-spectrograms
location = 'total_wav/'
# Name of the file with the song ids and respective quadrants
csv_name = 'new_pub_complete_targets.csv'
# Name of the file containing all samples. The program expects a pickle file
# with the structure: (num_samples, (song_id, waveform, target))
samples_file = 'new_pub_complete_samples_normalized_22kHz'
# Name of the file containing all resulting spectrograms of applying STFT on all samples.
# The program expects a pickle file with the structure: (num_of_samples, (width, height))
stft_file = "dataset_22kHz_stft_norm_fixed"
# Name of the file containing all Mel-spectrogram representations of the samples.
# The program expects a pickle file with the structure: (num_of_samples, (width, height))
mel_file = 'new_pub_complete_dataset_22kHz_melspect_norm'
# Name of the file containing all MFCC spectrogram representation of the samples.
# The program expects a pickle file with the structure: (num_of_samples, (width, height))
mfcc_file = "dataset_22kHz_mfcc_norm_fixed"
# FIles containing AV values calculated using Warriner's Affect Dictionary.
# Each word used in the original tags attributed to each song as a defined
# arousal, valence and dominance value in this dictionary. By averaging the values
# corresponding to all tags, the AV values for the song in question can be obtained.
file_with_median_av_values = 'av_warrine_new_pub_complete_all_median.csv' # 0 to 1 (arousal and valence ranges)
file_with_negative_av_values= 'av_warrine_new_pub_complete_all_negative.csv' # -1 to 1
# Name of the files containg the STFT, Mel and MFCC spectrograms obatined using a 22kHz
# sample rate.
file_with_mel_22kHz = 'dataset_22kHz_melspect_norm_fixed'
file_with_mfcc_22kHz = 'dataset_22kHz_mfccs_norm_fixed'
file_with_stft_22khz = 'dataset_22kHz_stft_norm_fixed'
# Name of the files containg the segmented STFT, Mel and MFCC spectrograms obatined using a 22kHz
# sample rate.
file_with_mel_22kHz_chunked = 'dataset_22kHz_melspect_norm_fixed_chunked'
file_with_mfcc_22kHz_chunked = 'dataset_22kHz_mfcc_norm_fixed_chunked'
file_with_stft_22khz_chunked = 'dataset_22kHz_stft_norm_fixed_chunked'
# The options below define the window size to chunk a given sample for the respective rep.
width_time_mel_mfcc, freq_bins_mel, freq_bins_mfcc= 216, 128, 20
width_time_stft, freq_bins_stft = 431, 512
# TODO: Add options for stft, mel and mfcc computations
####################
### Utils ##########
####################

# Function that collects all the file names of the songs and sorts them by Song_ID and returns
# them as an array
def get_file_names():
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

# Function that loads and returns a sorted array with all Song_IDs
def load_features_fixed():
    # carregar .csv com top 100
    data = pd.read_csv(
        curr_path + ptd + csv_name,
        sep=',', header=0)    # ordenar pelo song code (ascendente)
    print(data.head())
    data = data.sort_values(by='SongID')
    song_code = data.iloc[:, 0]
    return None, None, song_code.tolist()

# Function that generates the sample file for the currently selected dataset.
# The .wav files generated as an intermediate step are saved to the previously
# specified directory.
# The normalize flag will transform all values into the [-1, 1] interval.
# The downsample flag forces the .wav file to be samples at the default
# sampling rate for librosa, 22.05 kHz.
def load2mel_fixed_22kHz(normalize_flag, downsample_flag, file2save):
    # fname here is the complete path to the file
    target = pd.read_csv(curr_path + ptd + csv_name, sep=',', header=0)
    corrected_songs = target.iloc[:, 0].to_numpy()
    target = target.set_index('SongID')

    try:
        makedirs(curr_path + ptd + location)
    except FileExistsError:
        pass

    print(corrected_songs)
    acum = 0
    # .mp3
    files = get_file_names()
    files = [file for file in files if file.split('/')[-1][:-4] in corrected_songs]
    files = natsorted(list(set(files)))
    sounds = []
    for fname in tqdm(files):
        f_just_name = fname.split('/')[-1]
        if f_just_name[:-4] in corrected_songs:
            # translate to .wav and saves to
            sound_temp = AudioSegment.from_mp3(fname)
            wav_file_location = location + f_just_name[:-4] + '.wav'
            print(wav_file_location)
            if normalize_flag:
                # in case normalize
                sound_temp = AudioSegment.normalize(sound_temp)
            sound_temp.export(wav_file_location, format="wav")
            # save values to y

            # downsample to 16000:
            if downsample_flag:
                y, _ = librosa.load(wav_file_location, sr=22050)
            else:
                # leave it to the original freq
                y, _ = librosa.load(wav_file_location)
            # sounds = [filename, [freq], target]
            sounds.append([f_just_name[:-4], y, target.loc[f_just_name[:-4], 'Quadrant']])
            acum += 1

    # save to file
    with open(file2save + ".pickle", "wb") as sounds_file:
        pickle.dump(sounds, sounds_file)

    print("Done!")


# Translates from class labels to binary labels identifying the corresponding quadrant.
def quadrant2hemispheres(y):
    valence4voice = []
    arousal4normal = []
    for item in y:
        if item == 'Q1':
            valence4voice.append(1)
            arousal4normal.append(1)
        elif item == 'Q2':
            valence4voice.append(0)
            arousal4normal.append(1)
        elif item == 'Q3':
            valence4voice.append(0)
            arousal4normal.append(0)
        elif item == 'Q4':
            valence4voice.append(1)
            arousal4normal.append(0)
    return valence4voice, arousal4normal


# Translates AV values to quadrant
def av2quad(av_values, middle_point):
    real = []
    for i in range(len(av_values)):
        if av_values[i, 0] > middle_point:
            if av_values[i, 1] > middle_point:
                quad_real = 1
            else:
                quad_real = 2
        else:
            if av_values[i, 1] > middle_point:
                quad_real = 4
            else:
                quad_real = 3
        real.append(quad_real)
    return np.asarray(real) - 1

# This function creates the spectrogram result of applying STFT on all samples from the previously
# created samples file (see the load2mel_fixed function), and saves them to the previously
# specified directory.
# 30 seconds with a sr of 22050 generates a spectrogram of shape (mel_bins, 2586)
def create_stfts():
    width_time = 2586
    max_len = 661500

    with open(samples_file + '.pickle', 'rb') as sounds_file:
        ssounds = pickle.load(sounds_file)

    # Due to some of the samples from the new datasets being over 30 seconds, this step
    # ensures that the clips do not exceed this length.
    # The excess is removed from the end of the clip, since most of them have more relevant
    # information in the begining. Feel free to try removing from the begining if it makes sense
    # on your data.
    for count, sound in enumerate(ssounds):
        if len(sound[1]) > max_len:
            to_cut = len(sound[1]) - max_len
            ssounds[count][1] = sound[1][to_cut:]

    ssounds.sort(key=lambda ssounds: ssounds[0])

    print('STFT dataset loading...')
    stft_all = [librosa.power_to_db(librosa.stft(temp[1], n_fft=1028)) for temp in tqdm(ssounds)]
    print('STFT loaded!')

    stft_all_rotated = [np.rot90(np.pad(temp, ((0, 0), (0, width_time - len(temp[0]))))) for temp in stft_all]

    df_stft = np.array(stft_all_rotated)
    print("Saving stft into file")
    with open(stft_file + '.pickle', 'wb') as stft_ff:
        pickle.dump(df_stft, stft_ff)


# This function creates the Mel-spectrogram representations of all samples from the previously
# created samples file (see the load2mel_fixed function), and saves them to the previously
# specified directory.
# 30 seconds with a sr of 22050 generates a melsprectrogram of shape (mel_bins, 1289)
def create_melspectrograms(file_with_samples, file_with_mel):
    width_time = 1298
    max_len = 661500

    with open(file_with_samples + ".pickle", "rb") as sounds_file:
        sounds = pickle.load(sounds_file)
    sounds.sort(key=lambda sounds: sounds[0])

    for count, sound in enumerate(sounds):
        if len(sound[1]) > max_len:
            to_cut = len(sound[1]) - max_len
            sounds[count][1] = sound[1][to_cut:]

    print("Dataset a ser carregado")
    spect_all = [librosa.power_to_db(librosa.feature.melspectrogram(temp[1])) for temp in tqdm(sounds)]
    print("Dataset carregado com sucesso!")

    # padding line column and rotate to get the
    # format, such as:
    # [samples][width][height][channels]
    # padding to form 1296/942 (width_time) -> to be used in the CNN
    # it has to be uniform to be converted to a np.array
    spect_all_rotated = [np.rot90(np.pad(temp, ((0, 0), (0, width_time - len(temp[0]))))) for temp in tqdm(spect_all)]

    # PANDAS _> NUMPY
    df_spect = np.array(spect_all_rotated)
    print("Saving mel-spectograms into file...")
    with open(file_with_mel + ".pickle", "wb") as mel_file:
        pickle.dump(df_spect, mel_file)

# This function creates the Mel-spectrogram representations of all samples from the previously
# created samples file (see the load2mel_fixed function), and saves them to the previously
# specified directory.
# 30 seconds with a sr of 22050 generates a melsprectrogram of shape (mel_bins, 1289)
def create_mfccs():
    width_time = 1298
    max_len = 661500

    file2save = "new_pub_complete_samples_normalized_22kHz"
    filewithmfcc = "dataset_22kHz_mfccs_norm_fixed"

    with open(file2save + ".pickle", 'rb') as sounds_file:
        msounds = pickle.load(sounds_file)

    for count, sound in enumerate(msounds):
        if len(sound[1]) > max_len:
            to_cut = len(sound[1]) - max_len
            msounds[count][1] = sound[1][to_cut:]

    msounds.sort(key=lambda msounds: msounds[0])

    print('MFCC dataset a ser carregado')
    mfcc_all = [librosa.power_to_db(librosa.feature.mfcc(temp[1], sr=22050)) for temp in tqdm(msounds)]
    print('Done')
    mfcc_all_rotated = [np.rot90(np.pad(temp, ((0, 0), (0, width_time - len(temp[0]))))) for temp in mfcc_all]

    df_mfcc = np.array(mfcc_all_rotated)
    print("Saving mffcs into file")
    with open(filewithmfcc + ".pickle", "wb") as mel_file:
        pickle.dump(df_mfcc, mel_file)


def chunk_full_to_5s(file_with_mel, file_to_save, stft=False):
    # width_time = 94 # 3s for 16kHz sr and 128 mel_bands
    if stft:
        width_time = 431
    else:
        width_time = 216  # 3s for 22.05kHz sr and 128 mel_bands, as per the authors method
    num_chunks = 6

    with open(file_with_mel + '.pickle', 'rb') as mel_file:
        df_spect = pickle.load(mel_file)
    df_spect = np.asarray(df_spect)

    to_remove = df_spect.shape[1] % num_chunks
    trimmed = [samp[to_remove:] for samp in df_spect]

    chunked_songs = []
    for samp in trimmed:
        song_chunks = []
        for i in range(num_chunks):
            song_chunks.append(samp[i * width_time:(i + 1) * width_time])
        chunked_songs.append(song_chunks)

    chunked_songs = np.asarray(chunked_songs)

    with open(file_to_save + '.pickle', 'wb') as chunk_file:
        pickle.dump(chunked_songs, chunk_file)

    print('Done')


def avg(lst):
    return sum(lst) / len(lst)


def takeSongCode(item):
    return item[0]

# Function that returns the index of the results obtained from the fold nearest
# to the mean of the results of all folds for the training history
# (see end of run_f1score function for context).
def find_index_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Function that removes from the sounds input array all songs which Song_ID is not present in the
# input songs_code array, returning the resulting array of songs.
def filterout(sounds, songs_code):
    # filter out songs without features
    final_set = []
    for temp in sounds:
        # verify if the song has features
        if temp[0] in songs_code:
            final_set.append(temp)
    # sort to check if everything is alright
    final_set.sort(key=takeSongCode)
    print("Retirei " + str(len(sounds) - len(final_set)) + " musicas!")
    return final_set

# Creates necessary directories to save the relevant metrics for evaluation
# as well as backup, should the process be terminated early.
def create_dirs_for_model(candidate):
    try:
        makedirs('conf_matrix/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs('f1_macro_best100/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs('f1_each/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs('history/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs('scores/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    return

# Picks up on the files saved from a previous run and proceeds from
# were it left off.
def load_prev_data(bs, ep, lr, directory):
    hist_names = natsorted([f for f in listdir('history/' + directory) if isfile(join('history/' + directory, f))])
    f1_names = natsorted(
        [f for f in listdir('f1_macro_best100/' + directory) if isfile(join('f1_macro_best100/' + directory, f))])
    f1_each_names = natsorted([f for f in listdir('f1_each/' + directory) if isfile(join('f1_each/' + directory, f))])
    conf_names = natsorted(
        [f for f in listdir('conf_matrix/' + directory) if isfile(join('conf_matrix/' + directory, f))])
    scores_names = natsorted(
        [f for f in listdir('scores/' + directory) if isfile(join('scores/' + directory, f))])

    hist_av, hist_val_av, hist_loss, hist_val_loss = [], [], [], []
    f1_s, f1_e, confs, scores = [], [], [], []

    print('Loading history files...')
    for n in tqdm(hist_names):
        with open('history/' + directory + '/' + n, 'rb') as hist_file:
            temp_hist = pickle.load(hist_file)
            hist_av.append([temp_hist.rmse_valence, temp_hist.rmse_arousal])
            hist_val_av.append([temp_hist.val_rmse_valence, temp_hist.val_rmse_arousal])
            hist_loss.append(temp_hist.loss)
            hist_val_loss.append(temp_hist.val_loss)

    print('Loading f1 files...')
    for n in tqdm(f1_names):
        with open('f1_macro_best100/' + directory + '/' + n, 'rb') as f1_file:
            f1_s.append(pickle.load(f1_file))

    print('Loading f1 each files...')
    for n in tqdm(f1_each_names):
        with open('f1_each/' + directory + '/' + n, 'rb') as f1_each_file:
            f1_e.append(pickle.load(f1_each_file))

    print('Loading confusion matrix files...')
    for n in tqdm(conf_names):
        with open('conf_matrix/' + directory + '/' + n, 'rb') as conf_names:
            confs.append(pickle.load(conf_names))

    print('Loading scores files...')
    for n in tqdm(scores_names):
        with open('scores/' + directory + '/' + n, 'rb') as scores_names:
            scores.append(pickle.load(scores_names))

    with open('folds/folds_{}_{}_{}.pickle'.format(bs, ep, lr), 'rb') as folds_file:
        folds = pickle.load(folds_file)

    return [hist_av, hist_val_av, hist_loss, hist_val_loss, f1_s, f1_e, confs, folds, len(f1_names), scores]

# Calculate F1 Score for AV output
def f1_score_AV(x, av_values, model, middle_point):
    # middle_point _> para o caso de termos valores entre 0 e 1 e outro set com valores entre -1 e 1
    # testar modelo
    pred = []
    real = []
    av_real = []
    av_pred = []

    mel, mfcc, stft = x[0], x[1], x[2]
    for i in range(len(x[0])):
        (arousal, valence) = model.predict(
            [np.expand_dims(mel[i], axis=0),
             np.expand_dims(mfcc[i], axis=0),
             np.expand_dims(stft[i], axis=0)])
        # PRED
        if arousal > middle_point:
            if valence > middle_point:
                quad = 1
            else:
                quad = 2
        else:
            if valence > middle_point:
                quad = 4
            else:
                quad = 3
        pred.append(quad)
        av_pred.append([arousal.tolist(), valence.tolist()])
        # REAL
        if av_values[i, 0] > middle_point:
            if av_values[i, 1] > middle_point:
                quad_real = 1
            else:
                quad_real = 2
        else:
            if av_values[i, 1] > middle_point:
                quad_real = 4
            else:
                quad_real = 3
        real.append(quad_real)
        av_real.append([av_values[i, 0], av_values[i, 1]])

    # f1score - first 20
    print("Real:")
    print(av_real[0:20])
    print(real[0:20])
    print("Pred:")
    print(pred[0:20])
    print(av_pred[0:20])
    result = f1_score(real, pred, average='macro')
    print("Macro F1-Score: " + str(result))
    f1_each = f1_score(real, pred, average=None)
    print("F1-Score: " + str(f1_each))
    return result, np.asarray(pred) - 1, f1_each, np.asarray(real) - 1


def display_stft(samp):
    fig, axs = plt.subplots(1, 1)
    axs = librosa.display.specshow(samp)
    plt.show()


def display_mel(samp):
    fig, axs = plt.subplots(1, 1)
    axs = librosa.display.specshow(samp)
    plt.show()


def display_mfcc(samp):
    fig, axs = plt.subplots(1, 1)
    axs = librosa.display.specshow(samp)
    plt.show()


def visualize_reps(filewav, filemel, filemfcc, filestft):
    with open(filewav + '.pickle', 'rb') as wavf:
        wav_reps = pickle.load(wavf)
        wav_reps.sort(key=lambda wav_reps: wav_reps[0])
        wav_reps = [wav_rep[1] for wav_rep in wav_reps]
    with open(filemel + '.pickle', 'rb') as melf:
        mel_reps = pickle.load(melf)
    with open(filemfcc + '.pickle', 'rb') as mfccf:
        mfcc_reps = pickle.load(mfccf)
    with open(filestft + '.pickle', 'rb') as stftf:
        stft_reps = pickle.load(stftf)

    to_show = np.random.choice(range(len(mel_reps)), 1)[0]

    #pyplot.subplot(2, 2, 1)
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(y=wav_reps[to_show], sr=16000)
    plt.subplot(3, 1, 2)
    librosa.display.waveplot(y=wav_reps[to_show], sr=16000)
    plt.subplot(3, 1, 3)
    librosa.display.waveplot(y=wav_reps[to_show], sr=16000)
    plt.show()

    #pyplot.show()
    display_stft(np.rot90(stft_reps[to_show], k=-1))
    display_mel(np.rot90(mel_reps[to_show], k=-1))
    display_mfcc(np.rot90(mfcc_reps[to_show], k=-1))

###############################################
## Models ######################################
###############################################
#############################
### Convolutional 2D Block #######
#############################
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, for_name, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(out_channels, kernel_size=(3, 3), strides=(1, 1),
                           data_format='channels_last', padding='same',
                           name='conv_{}'.format(for_name))
        self.mp = MaxPooling2D(name='mp_{}'.format(for_name))
        self.dp = Dropout(0.2, name='dp_{}'.format(for_name))
        self.relu = ReLU(name='relu_{}'.format(for_name))

    def call(self, input):
        out = self.conv(input)
        out = self.mp(out)
        out = self.dp(out)
        out = self.relu(out)
        print(out.shape)
        return out

####################################
### Feature extractor for STFT reps #######
####################################
def cnn_for_stft(lr=0.001):
    model_stft = Input(shape=(width_time_stft, freq_bins_stft, 1))

    out_mfcc = ConvBlock('stft_1', 16)(model_stft)
    out_mfcc = ConvBlock('stft_2', 32)(out_mfcc)
    out_mfcc = ConvBlock('stft_3', 64)(out_mfcc)
    out_mfcc = ConvBlock('stft_4', 128)(out_mfcc)
    out_mfcc = ConvBlock('stft_5', 256)(out_mfcc)
    out_mfcc = ConvBlock('stft_6', 256)(out_mfcc)

    out_mfcc = AdaptiveAveragePooling2D(output_size=(1, 2))(out_mfcc)
    out_mfcc = Flatten()(out_mfcc)

    return Model(model_stft, out_mfcc)

###############################################
### Feature extractor for Mel-spectrograms reps #######
###############################################
def cnn_for_mel(lr=0.001):
    model_mel = Input(shape=(width_time_mel_mfcc, freq_bins_mel, 1))

    out_mel = ConvBlock('mel_1', 16)(model_mel)
    out_mel = ConvBlock('mel_2', 32)(out_mel)
    out_mel = ConvBlock('mel_3', 64)(out_mel)
    out_mel = ConvBlock('mel_4', 128)(out_mel)
    out_mel = ConvBlock('mel_5', 256)(out_mel)

    out_mel = AdaptiveAveragePooling2D(output_size=(1, 2))(out_mel)
    out_mel = Flatten()(out_mel)

    return Model(model_mel, out_mel)

###############################################
### Feature extractor for MFCC reps #################
###############################################
def cnn_for_mfcc(lr=0.001):
    model_mfcc = Input(shape=(width_time_mel_mfcc, freq_bins_mfcc, 1))

    out_mfcc = ConvBlock('mfcc_1', 16)(model_mfcc)
    out_mfcc = ConvBlock('mfcc_2', 32)(out_mfcc)
    out_mfcc = ConvBlock('mfcc_3', 64)(out_mfcc)

    out_mfcc = AdaptiveAveragePooling2D(output_size=(1, 2))(out_mfcc)
    out_mfcc = Flatten()(out_mfcc)

    return Model(model_mfcc, out_mfcc)

###############################################
### Complete MSR Segment-level model ###############
###############################################
def multi_rep_model(learning_rate=0.01):
    mel_model = cnn_for_mel(learning_rate)
    mfcc_model = cnn_for_mfcc(learning_rate)
    stft_model = cnn_for_stft(learning_rate)

    model_comb = concatenate([mel_model.layers[-1].output,
                              mfcc_model.layers[-1].output,
                              stft_model.layers[-1].output])

    model_comb = Dense(512, name='dense_final_1')(model_comb)
    model_comb = Dropout(0.2)(model_comb)
    model_comb = ReLU()(model_comb)

    model_comb = Dense(128, name='dense_final_2')(model_comb)
    model_comb = ReLU()(model_comb)

    # Sigmoid for median values
    model_comb_1 = Dense(1, activation='tanh', name="a")(model_comb)
    model_comb_2 = Dense(1, activation='tanh', name="v")(model_comb)

    another_model = Model(inputs=[mel_model.input,
                                  mfcc_model.input,
                                  stft_model.input],
                          outputs=[model_comb_1,
                                   model_comb_2])
    opt = SGD(learning_rate)
    another_model.compile(loss={'a': 'mse', 'v': 'mse'}, optimizer=opt,
                          metrics={'a': ['mean_absolute_error', RootMeanSquaredError()],
                                   'v': ['mean_absolute_error', RootMeanSquaredError()]})
    another_model.summary()
    return another_model

###############################################
## Model training #############################
## and ########################################
## testing ####################################
###############################################

# Trains and tests the model for a certain set of [epochs, batch_size, learning_rate],
# using cross-validation to get 100 different train/test folds to more accurately
# assess the model's performance.
def run_f1score(X, labels, labels_av, epochs, batch_size, learning_rate, resume_training=False):
    print("começo a treinar:")
    # F1 Score with Repeated Stratified K Fold, 10 splits, 10 repeats, resulting
    # in 100 train/test folds
    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    acum, loss, scores_all = 0, [], []

    history_rmse_valence = []
    history_rmse_arousal = []
    history_rmse_valence_val = []
    history_rmse_arousal_val = []
    history_all_loss = []
    history_all_loss_val = []

    # Segment-level results (per clip == per segment)
    macro_f1_score_per_clip, f1_each_per_clip, confusion_matrix_global_per_clip = [], [], np.zeros((4, 4))

    # Sample-level results
    macro_f1_score, f1_each, confusion_matrix_global = [], [], np.zeros((4, 4))

    # save the model with the highest f1_score
    max_f1, max_f1_per_clip = 0, 0

    # Get STFT, Mel and MFCC representations for each sample
    mel_samps = X[0]
    mffc_samps = X[1]
    stft_samps = X[2]

    arousal = labels_av[:, 0]
    valence = labels_av[:, 1]

    a_loss, v_loss = [], []

    name_of_dir = 'bs_{}_ep_{}_lr_{}/'.format(batch_size, epochs, learning_rate)

    ###############################################
    ## Callbacks ####################################
    ###############################################
    # A custom early stopping strategy was used here. Define how many epochs (patience)
    # may the strategy endure without improvements to the baseline (loss) before shutting
    # down training.
    es = MyCustomEarlyStopping(patience=10, baseline=5)
    ###############################################

    folds = list(kfold.split(X[0], labels))
    folds_name = 'folds_' + str(batch_size) + '_' + str(epochs) + '_' + str(learning_rate)

    if resume_training:
        prev_data = load_prev_data(batch_size, epochs, learning_rate,
                                   directory='bs_{}_ep_{}_lr_{}'.format(batch_size, epochs, learning_rate))

        history_rmse_valence = [av[0] for av in prev_data[0]]
        history_rmse_arousal = [av[1] for av in prev_data[0]]
        history_rmse_valence_val = [av[0] for av in prev_data[1]]
        history_rmse_arousal_val = [av[1] for av in prev_data[1]]
        history_all_loss = prev_data[2]
        history_all_loss_val = prev_data[3]

        macro_f1_score = prev_data[4]

        f1_each = prev_data[5]

        for conf in prev_data[6]:
            confusion_matrix_global += conf

        acum = prev_data[8]

        if acum < 100:
            folds = prev_data[7][acum:]
        else:
            folds = []

        scores_all = prev_data[9]

        print('Resume at {} fold...'.format(acum))

    else:
        with open('folds/' + folds_name + '.pickle', 'wb') as file_with_folds:
            pickle.dump(folds, file_with_folds)

    # Loop for train/test while having folds:
    for train, test in tqdm(folds):
        acum += 1

        ## Training portion ##
        print('Training ' + str(acum) + '...')

        print('Clearing prev model.')
        clear_session()
        # re-initilize and compile
        model = multi_rep_model(learning_rate)

        x_train_mel, x_train_mfcc, x_train_stft, y_a_train, y_v_train = \
            [], [], [], [], []
        for i in train:
            for temp_mel, temp_mfcc, temp_stft in zip(mel_samps[i], mffc_samps[i], stft_samps[i]):
                x_train_mel.append(temp_mel)
                x_train_mfcc.append(temp_mfcc)
                x_train_stft.append(temp_stft)
                y_a_train.append(arousal[i])
                y_v_train.append(valence[i])

        # Width time for 5 seconds, height same as full length samples
        x_train_mel = np.array(x_train_mel).reshape(len(x_train_mel), width_time_mel_mfcc, 128, 1)
        x_train_mfcc = np.array(x_train_mfcc).reshape(len(x_train_mfcc), width_time_mel_mfcc, 20, 1)
        x_train_stft = np.array(x_train_stft).reshape(len(x_train_stft), width_time_stft, 515, 1)

        x_test_mel, x_test_mfcc, x_test_stft, y_a_test, y_v_test = \
            [], [], [], [], []
        x_test_mel_chunk, x_test_mfcc_chunk, x_test_stft_chunk, y_a_test_chunk, y_v_test_chunk = \
            [], [], [], [], []

        for j in test:
            for temp_mel, temp_mfcc, temp_stft in zip(mel_samps[j], mffc_samps[j], stft_samps[j]):
                x_test_mel_chunk.append(temp_mel)
                x_test_mfcc_chunk.append(temp_mfcc)
                x_test_stft_chunk.append(temp_stft)
                y_a_test_chunk.append(arousal[j])
                y_v_test_chunk.append(valence[j])
            x_test_mel.append(np.array(mel_samps[j]).reshape(len(mel_samps[j]), width_time_mel_mfcc, 128, 1))
            x_test_mfcc.append(np.array(mffc_samps[j]).reshape(len(mffc_samps[j]), width_time_mel_mfcc, 20, 1))
            x_test_stft.append(np.array(stft_samps[j]).reshape(len(stft_samps[j]), width_time_stft, 515, 1))
            y_a_test.append(arousal[j])
            y_v_test.append(valence[j])

        x_test_mel_chunk = np.array(x_test_mel_chunk).reshape(len(x_test_mel_chunk), width_time_mel_mfcc, 128, 1)
        x_test_mfcc_chunk = np.array(x_test_mfcc_chunk).reshape(len(x_test_mfcc_chunk), width_time_mel_mfcc, 20, 1)
        x_test_stft_chunk = np.array(x_test_stft_chunk).reshape(len(x_test_stft_chunk), width_time_stft, 515, 1)

        # train and save to log file
        history = model.fit([x_train_mel, x_train_mfcc, x_train_stft],
                            [y_a_train, y_v_train],
                            validation_data=([x_test_mel_chunk,
                                              x_test_mfcc_chunk,
                                              x_test_stft_chunk],
                                             [y_a_test_chunk, y_v_test_chunk]),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[es], verbose=1)

        print("\nArousal loss on last epoch:")
        print(history.history['a_loss'][-1])
        print("\nValence loss on last epoch:")
        print(history.history['v_loss'][-1])
        a_loss.append(history.history['a_loss'][-1])
        v_loss.append(history.history['v_loss'][-1])

        # Save accuracy and loss from the training phase
        history_rmse_valence.append(history.history['v_root_mean_squared_error'])
        history_rmse_valence_val.append(history.history['val_v_root_mean_squared_error'])
        history_rmse_arousal.append(history.history['a_root_mean_squared_error'])
        history_rmse_arousal_val.append(history.history['val_a_root_mean_squared_error'])
        history_all_loss.append(history.history['loss'])
        history_all_loss_val.append(history.history['val_loss'])

        ## Testing portion ##
        print('Testing ' + str(acum) + '...')
        # Evaluate the model
        score_temp = model.evaluate(x=[x_test_mel_chunk, x_test_mfcc_chunk, x_test_stft_chunk],
                                        y=[y_a_test_chunk, y_v_test_chunk],
                                        verbose=0)
        print(score_temp)
        scores_all.append(score_temp)

        ## Segment-level results ##
        ypred_all = model.predict(x=X[test])

        # Calculate Confusion Matrix
        print(confusion_matrix(np.asarray(labels[test]).argmax(axis=1), np.asarray(ypred_all).argmax(axis=1)))
        conf_temp_per_clip = confusion_matrix(np.asarray(labels[test]).argmax(axis=1), np.asarray(ypred_all).argmax(axis=1))
        confusion_matrix_global_per_clip = confusion_matrix_global + conf_temp_per_clip

        # Predict for F1 Score
        f1_temp_per_clip = f1_score(np.asarray(labels[test]).argmax(axis=1), np.asarray(ypred_all).argmax(axis=1), average='macro')
        f1_each_per_clip.append(f1_score(np.asarray(labels[test]).argmax(axis=1), np.asarray(ypred_all).argmax(axis=1), average=None))
        print("\nF1_Score: " + str(f1_temp_per_clip))
        # FIXME: CHANGE HERE
        if f1_temp_per_clip > max_f1_per_clip:
            max_f1_per_clip = f1_temp_per_clip
            best_acum_per_clip = acum
        macro_f1_score_per_clip.append(f1_temp_per_clip)

        ## Sample-level results ##
        y_mean_test_all = []

        for test_sample_mel, test_sample_mfcc, test_sample_stft, y_a, y_v in \
                zip(x_test_mel, x_test_mfcc, x_test_stft, y_a_test, y_v_test):
            ypred = model.predict_on_batch(x=[
                test_sample_mel, test_sample_mfcc, test_sample_stft])
            ypred_conc = []
            for y_a, y_v in zip(ypred[0], ypred[1]):
                ypred_conc.append([y_a.tolist()[0], y_v.tolist()[0]])
            # 0.5 since we are using av median values
            ypred_quad = av2quad(np.asarray(ypred_conc), 0)
            y_mean = np.around(np.mean(ypred_quad))
            y_mean_test_all.append(y_mean)

        # Get real labels
        test_av = []
        for y_a, y_v in zip(y_a_test, y_v_test):
            test_av.append([y_a, y_v])
        test_quad = av2quad(np.asarray(test_av), 0)

        conf_temp = confusion_matrix(np.asarray(test_quad), np.asarray(y_mean_test_all))
        print(conf_temp)
        confusion_matrix_global += conf_temp

        f1_temp = f1_score(np.asarray(test_quad),
                           np.asarray(y_mean_test_all),
                           average='macro')
        f1_each.append(f1_score(np.asarray(test_quad),
                                np.asarray(y_mean_test_all),
                                average=None))

        if f1_temp > max_f1:
            max_f1 = f1_temp
        macro_f1_score.append(f1_temp)

        # Save metrics from the current test fold
        with open('history/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_hist:
            hist_acum = History(history.history['loss'], history.history['val_loss'],
                                           history.history['v_root_mean_squared_error'],
                                           history.history['val_v_root_mean_squared_error'],
                                           history.history['a_root_mean_squared_error'],
                                           history.history['val_a_root_mean_squared_error'])
            pickle.dump(hist_acum, file_hist)

        with open('f1_macro_best100_perclip/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_f1_pc:
            pickle.dump(f1_temp_per_clip, file_f1_pc)
        with open('f1_macro_best100/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_f1:
            pickle.dump(f1_temp, file_f1)

        with open('f1_each_perclip/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_each_pc:
            pickle.dump(f1_each_per_clip[-1], file_each_pc)
        with open('f1_each/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_each:
            pickle.dump(f1_each[-1], file_each)

        with open('conf_matrix_perclip/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_conf_pc:
            pickle.dump(conf_temp_per_clip, file_conf_pc)
        with open('conf_matrix/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_conf:
            pickle.dump(conf_temp, file_conf)

        with open('scores/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_scores:
            pickle.dump(scores_all[-1], file_scores)

        # This code block intends to completely wipe the model trained in the current
        # loop iteration, since it is no longer necessary at this point, to save memory
        # on the CPU/GPU and prevent an eventual crash
        del model
        gc.collect()
        clear_session()
        tf.compat.v1.reset_default_graph()

        continue

    # Print out relavant info from the train/test process
    print("epochs:\t" + str(epochs))
    print("batch:\t" + str(batch_size))
    print("Results: ")
    print(scores_all)
    print("\nF1-Score (ALL): ")
    print(macro_f1_score)
    print("MEAN F1-SCORE:")
    print(avg(macro_f1_score))
    print("STD F1-SCORE:")
    print(np.asarray(macro_f1_score).std())
    print("MEAN F1-SCORE PER CLASS:")
    print("Q1\tQ2\tQ3\tQ4")
    print(str(avg(np.asarray(f1_each)[:, 0])) + '\t' + str(avg(np.asarray(f1_each)[:, 1])) + '\t' + str(
        avg(np.asarray(f1_each)[:, 2])) + '\t' + str(avg(np.asarray(f1_each)[:, 3])))
    print("CONFUSION MATRIX:")
    print(confusion_matrix_global)
    print("MEAN LOSS:")
    print(avg(np.asarray(scores_all)[:, 0]))
    print("STD LOSS:")
    print(np.asarray(np.asarray(scores_all)[:, 0]).std())
    print("MEAN RMSE:")
    print("AROUSAL:")
    print(avg(np.asarray(scores_all)[:, 4]))
    print("VALENCE")
    print(avg(np.asarray(scores_all)[:, 6]))
    print("STD RMSE:")
    print("AROUSAL:")
    print(np.asarray(np.asarray(scores_all)[:, 4]).std())
    print("VALENCE")
    print(np.asarray(np.asarray(scores_all)[:, 6]).std())

    # save results
    # get the average f1 score
    idx = find_index_nearest(macro_f1_score, avg(macro_f1_score))
    hist = History(history_all_loss[idx],
                              history_all_loss_val[idx],
                              history_rmse_valence[idx],
                              history_rmse_valence_val[idx],
                              history_rmse_arousal[idx],
                              history_rmse_arousal_val[idx])

    return hist, macro_f1_score, f1_each, confusion_matrix_global, scores_all, \
            macro_f1_score_per_clip, f1_each_per_clip, confusion_matrix_global_per_clip

# Main function used for optimizing the model at hand.
def my_gridsearch_simple(x, Y, Y_av):
    # Create the necessary directories to save files for each fold iteration
    try:
        makedirs("f1_macro_best100_perclip/")
    except FileExistsError:
        pass
    try:
        makedirs("f1_macro_best100/")
    except FileExistsError:
        pass

    try:
        makedirs("f1_each_perclip/")
    except FileExistsError:
        pass
    try:
        makedirs("f1_each/")
    except FileExistsError:
        pass

    try:
        makedirs("history/")
    except FileExistsError:
        pass

    try:
        makedirs("conf_matrix_perclip/")
    except FileExistsError:
        pass
    try:
        makedirs("conf_matrix/")
    except FileExistsError:
        pass

    try:
        makedirs("folds/")
    except:
        pass

    # Permutations between all values on this dictionary are experimented with.
    # Leave unchanged for replicating the experiment.
    # Change these or add more for experimentation.
    # To experiment with other optimizers, change this in the function for creating the model.
    param_grid = {
        'batch_size': [16],  # removed 16
        'epochs': [50],
        'learning_rate': [0.001] #0.01 ???
    }

    # When the current permutation of the previous values match a set of values
    # [batch_size, epochs, learning_rate] from below, the training phase is resumed,
    # continuing from the first fold without any related files.
    # If a set of values was not ran previously, this will crash.
    resume_params_sgd = [
    ]

    # Each set of values [batch_size, epochs, learning_rate] are skipped.
    ran_params_sgd = [
    ]

    names_of_params = param_grid.keys()
    matrix_of_params = []
    for i in range(len(param_grid)):
        matrix_of_params = [param_grid[i] for i in names_of_params]

    start_grid_search = time.time()
    num_candidates = len(matrix_of_params[0]) * len(matrix_of_params[1]) * len(matrix_of_params[2])

    gridsearch_progress = IncrementalBar("Start GridSearch...", max=num_candidates)
    for bs in range(len(matrix_of_params[0])):
        for ep in range(len(matrix_of_params[1])):
            for lr in range(len(matrix_of_params[2])):
                curr_candidate = [matrix_of_params[0][bs], matrix_of_params[1][ep], matrix_of_params[2][lr]]

                curr_time_candidate = time.time()

                if curr_candidate in resume_params_sgd:
                    print('Resuming ' + str(curr_candidate) + '.')

                    hist, f1s, f1e, conf, scores, \
                    f1s_pc, f1e_pc, conf_pc = run_f1score(x, Y, Y_av,
                                                       batch_size=matrix_of_params[0][bs],
                                                       epochs=matrix_of_params[1][ep],
                                                       learning_rate=matrix_of_params[2][lr],
                                                       resume_training=True)

                elif curr_candidate in ran_params_sgd:
                    print(str(curr_candidate) + ' already tested.')
                    gridsearch_progress.next()
                    continue

                else:
                    create_dirs_for_model(curr_candidate)

                    hist, f1s, f1e, conf, scores, \
                    f1s_pc, f1e_pc, conf_pc = run_f1score(x, Y, Y_av,
                                                       batch_size=matrix_of_params[0][bs],
                                                       epochs=matrix_of_params[1][ep],
                                                       learning_rate=matrix_of_params[2][lr])

                file_name = 'history_msr_segment_bs_' + str(matrix_of_params[0][bs]) + \
                            '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1s_name = 'f1_score_msr_segment_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1s_pc_name = 'f1_score_perclip_msr_segment_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1e_name = 'f1_each_msr_segment_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1e_pc_name = 'f1_each_perclip_msr_segment_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                conf_name = 'conf_mat_msr_segment_bs' + str(matrix_of_params[0][bs]) + \
                            '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                conf_pc_name = 'conf_mat_perclip_msr_segment_bs' + str(matrix_of_params[0][bs]) + \
                            '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                scores_name = 'scores_msr_segment_bs' + str(matrix_of_params[0][bs]) + \
                              '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                time_name = 'time_msr_segment_bs' + str(matrix_of_params[0][bs]) + \
                            '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])

                with open('history/' + file_name + '.pickle', 'wb') as file_with_hist:
                    pickle.dump(hist, file_with_hist)

                with open('f1_macro_best100_perclip/' + f1s_pc_name + '.pickle', 'wb') as file_with_f1_pc:
                    pickle.dump(f1s_pc, file_with_f1_pc)
                with open('f1_macro_best100/' + f1s_name + '.pickle', 'wb') as file_with_f1:
                    pickle.dump(f1s, file_with_f1)

                with open('f1_each_perclip/' + f1e_pc_name + '.pickle', 'wb') as file_with_each_pc:
                    pickle.dump(f1e_pc, file_with_each_pc)
                with open('f1_each/' + f1e_name + '.pickle', 'wb') as file_with_each:
                    pickle.dump(f1e, file_with_each)

                with open('conf_matrix_perclip/' + conf_pc_name + '.pickle', 'wb') as file_with_conf_pc:
                    pickle.dump(conf_pc, file_with_conf_pc)
                with open('conf_matrix/' + conf_name + '.pickle', 'wb') as file_with_conf:
                    pickle.dump(conf, file_with_conf)

                with open('scores/' + scores_name + '.pickle', 'wb') as file_with_scores:
                    pickle.dump(scores, file_with_scores)

                curr_grid_search = time.time()

                time_curr = round(curr_grid_search - curr_time_candidate)

                with open(time_name, 'w') as time_file:
                    time_file.write('Ran for {} minutes.'.format(time_curr/ 60))

                time_elapsed = round(curr_grid_search - start_grid_search)
                print("GridSearch has run for {} minutes.".format(time_elapsed / 60))

                gridsearch_progress.next()

    end_grid_search = time.time()
    time_elapsed = round(end_grid_search - start_grid_search)
    print("GridSearch took {} minutes.".format(time_elapsed / 60))
    print("GridSearch complete.")
    return

def main(file_mel, file_mffc, file_stft, file_with_labels):
    with open(file_mel + '.pickle', 'rb') as mel_file:
        df_spect = pickle.load(mel_file)
    with open(file_mffc + '.pickle', 'rb') as mffc_file:
        df_mfcc = pickle.load(mffc_file)
    with open(file_stft + '.pickle', 'rb') as stft_file:
        df_stft = pickle.load(stft_file)
    with open(file_with_labels + '.pickle', 'rb') as sounds_file:
        sounds = pickle.load(sounds_file)

    # AV values instead of labels
    # 0 / 1
    file_av_median = pd.read_csv(curr_path + ptd + file_with_median_av_values, delimiter=',')
    # -1 / 1
    file_av = pd.read_csv(curr_path + ptd + file_with_negative_av_values, delimiter=',', header=0)
    file_songs = pd.read_csv(curr_path + ptd + csv_name, delimiter=',', header=0)

    file_songs_ids = file_songs['SongID'].tolist()
    file_av = file_av.loc[file_av['Song'].isin(file_songs_ids)]
    av_values = file_av.iloc[:, 1:]
    av_values = np.array(av_values[['Arousal','Valence']].values.tolist())

    file_av_median = file_av_median.loc[file_av_median['Song'].isin(file_songs_ids)]
    songs_with_av = file_av_median['Song'].tolist()
    av_values_median = file_av_median.iloc[:, 1:]
    av_values_median = np.array(av_values_median[['Arousal','Valence']].values.tolist())
    av_values_median = av_values_median * 2 - 1

    sounds.sort(key=lambda sounds: sounds[0])
    sounds_with_av = np.array([sound for sound in sounds if sound[0] in songs_with_av])
    index_to_keep = np.array([i for i, sound in enumerate(sounds)
                              if sound[0] in sounds_with_av])
    labels = np.array([temp[2] for temp in sounds])

    X_mel = df_spect.reshape(df_spect.shape[0], df_spect.shape[1],
                             df_spect.shape[2], df_spect.shape[3], 1)[index_to_keep]
    X_mfcc = df_mfcc.reshape(df_mfcc.shape[0], df_mfcc.shape[1],
                             df_mfcc.shape[2], df_mfcc.shape[3], 1)[index_to_keep]
    X_stft = df_stft.reshape(df_stft.shape[0], df_stft.shape[1],
                             df_stft.shape[2], df_stft.shape[3], 1)[index_to_keep]
    X = [X_mel, X_mfcc, X_stft]
    print(X_mel.shape)
    print(X_mfcc.shape)
    print(X_stft.shape)

    # converter para inteiros
    mapping = {}
    classes = ['Q1', 'Q2', 'Q3', 'Q4']
    for x in range(len(classes)):
        mapping[classes[x]] = x
    Y_total = np.array([mapping[temp] for temp in labels])[index_to_keep]

    # Y_total = np_utils.to_categorical(Y_total)

    print(Y_total[0])

    # verify quadrant
    quad_median = av2quad(av_values_median, middle_point=0)
    quad_avg = av2quad(av_values, middle_point=0)

    # Check the accuracy of the av values against the class labels. Class labels
    # should always be the most accurate targets.
    print(accuracy_score(Y_total, quad_median))
    print(accuracy_score(Y_total, quad_avg))

    print("Preprocessamento completo!")

    start = time.time()

    my_gridsearch_simple(X, Y_total, av_values_median)

    end = time.time()
    elapsed = round(end - start)
    print("Minutos passados: " + str(elapsed / 60))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    full_path = curr_path + ptd

    #load2mel_fixed_22kHz(True, True, full_path + samples_file)

    #create_melspectrograms()
    #create_mfccs()
    #create_stfts()

    #chunk_full_to_5s(full_path + file_with_mel_22kHz,
    #                                   full_path + file_with_mel_22kHz_chunked)
    #chunk_full_to_5s(full_path + file_with_mfcc_22kHz,
    #                                    full_path + file_with_mfcc_22kHz_chunked)
    #chunk_full_to_5s(full_path + file_with_stft_22khz,
    #                                   full_path + file_with_stft_22khz_chunked, stft=True)

    #visualize_reps(full_path + file_with_samples, full_path + file_with_mel,
    #               full_path + file_with_mfcc, full_path + file_with_stft)

    main(full_path + file_with_mel_22kHz_chunked,
                full_path + file_with_mfcc_22kHz_chunked,
                full_path + file_with_stft_22khz_chunked,
                full_path + samples_file)

    print('Done')
