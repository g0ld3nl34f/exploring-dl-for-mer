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
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalMaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.backend import clear_session
from pydub import AudioSegment
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
# spleeter conflict with llvmlite updated version
from progress.bar import IncrementalBar
import warnings
from pathlib import Path
import gc
from natsort import natsorted
from tqdm import tqdm
from tensorflow.python.client import device_lib

warnings.simplefilter("ignore")

from MyCallbacks import MyCallbacks
import History
from Residual2D import Residual2D

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
# TODO: Change to chunk files
samples_file = 'new_pub_complete_samples_normalized_16kHz'
# Name of the file containing all Mel-spectrogram representations of the samples.
# The program expects a pickle file with the structure: (num_of_samples, (width, height))
mel_file = 'new_pub_complete_dataset_16kHz_melspect_norm'
# Width for separating samples into smaller windows, or chunks
# Default is for a 16kHz sample rate and 128 frequency bin Mel-spectrogram
window_width = 116  # 3.69s
# This variable should be changed to the number of samples available
# in the dataset in use.
number_of_samples = 1629


####################
### Utils ############
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
        sep=',', header=0)  # ordenar pelo song code (ascendente)
    print(data.head())
    data = data.sort_values(by='SongID')
    song_code = data.iloc[:, 0]
    return None, None, song_code.tolist()


# Function that generates the sample file for the currently selected dataset.
# The .wav files generated as an intermediate step are saved to the previously
# specified directory.
# The normalize flag will transform all values into the [-1, 1] interval.
# The downsample flag downsamples the .wav file from the default 22.05kHz of librosa
# to 16kHz.
def load2mel_fixed(normalize_flag, downsample_flag, file2save):
    # fname here is the complete path to the file
    # anotacoes para adicionar - dicionario com nome e target
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
    sounds = []
    for fname in files:
        f_just_name = fname.split('/')[-1]

        if f_just_name[:-4] in corrected_songs:
            # translate to .wav and saves to
            sound_temp = AudioSegment.from_mp3(fname)
            wav_file_location = curr_path + ptd + location + f_just_name[:-4] + '.wav'
            print(wav_file_location)

            if normalize_flag:
                # in case normalize
                sound_temp = AudioSegment.normalize(sound_temp)
            sound_temp.export(wav_file_location, format="wav")
            # save values to y

            # downsample to 16000:
            if downsample_flag:
                y, _ = librosa.load(wav_file_location, sr=16000)
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


# This function creates the Mel-spectrogram representations of all samples from the previously
# created samples file (see the load2mel_fixed function), and saves them to the previously
# specified directory.
# 30 seconds with a sr of 16000 generates a melsprectrogram of shape (mel_bins, 938)
def create_melspectrograms(file_with_samples, file_with_mel):
    width_time = 942
    max_len = 482000

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


# Function to separate each waveform sample in chunks of ~3.5s with
# the following structure: (num_samples, (song_id, (num_chunks,(waveforms)), target))
def split_chunks_3_5s(file_with_music, file_chunked='fixed_chunked_sep_norm_16kHz_samples'):
    # This value represents ~3.5 seconds of a song with a sampling rate of 16kHz
    # 59049 timesteps for the raw waveform at 16kHz, duration ~3.5s
    duration_3_5 = 59049

    with open(file_with_music + '.pickle', 'rb') as sounds_file:
        songs = pickle.load(sounds_file)

    chunked_file = []
    all_chunks = 0
    chunking_process = IncrementalBar('Chunking {} songs...'.format(len(songs)), max=len(songs))
    for song in songs:
        chunked_song = []
        # Get time series array of the current song
        full_song = song[1]
        # Get number of full chunks in the given song
        num_chunks = len(full_song) // duration_3_5

        for chunk in range(num_chunks):
            chunked_song.append(np.array(full_song[duration_3_5 * chunk:duration_3_5 * (chunk + 1)]))

        chunked_file.append([song[0], chunked_song, song[2]])
        all_chunks += len(chunked_song)
        chunking_process.next()

    chunked_file = np.asarray(chunked_file)
    with open(file_chunked + ".pickle", "wb") as samples_file:
        pickle.dump(chunked_file, samples_file, protocol=4)


def samples_3_5s_to_mel(fws, fwm):
    width_time = 116  # ?

    with open(fws + '.pickle', 'rb') as file_with_samples:
        samps = pickle.load(file_with_samples)

    samps = natsorted(samps, key=lambda samps: samps[0])

    spect_all = []

    # Transform each chunk in each sample to a mel spectrogram
    # Preserve shape of file -> (num_songs, (song_code, (num_chunks), target/quadrant))
    print("Dataset a ser carregado")
    for samp in tqdm(samps):
        spect_song = []
        for chunk in samp[1]:
            spect_song.append(chunk)
        spect_song_mel = [librosa.power_to_db(librosa.feature.melspectrogram(temp)) for temp in spect_song]
        spect_all.append([np.rot90(temp) for temp in spect_song_mel])
    print("Dataset carregado com sucesso!")

    with open(fwm + '.pickle', 'wb') as file_with_mel:
        pickle.dump(spect_all, file_with_mel)


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
        makedirs('conf_matrix_perclip/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs('f1_macro_best100/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass
    try:
        makedirs('f1_macro_best100_perclip/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass

    try:
        makedirs('f1_each/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
    except FileExistsError:
        pass
    try:
        makedirs('f1_each_perclip/bs_{}_ep_{}_lr_{}/'.format(candidate[0], candidate[1], candidate[2]))
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

    hist_acc, hist_val_acc, hist_loss, hist_val_loss = [], [], [], []
    f1_s, f1_e, confs, scores = [], [], [], []

    print('Loading history files...')
    for n in tqdm(hist_names):
        with open('history/' + directory + '/' + n, 'rb') as hist_file:
            temp_hist = pickle.load(hist_file)
            hist_acc.append(temp_hist.acc)
            hist_val_acc.append(temp_hist.val_acc)
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

    return [hist_acc, hist_val_acc, hist_loss, hist_val_loss, f1_s, f1_e, confs, folds, len(f1_names), scores]


###############################################
## Model ######################################
###############################################
########################################################
### ShortChunk Model ###################################
### Original Model implemented in PyTorch by Won et al.##################################
### https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py ##
#########################################################################################
def compile_model(learning_rate=0.0001):
    # width_time = 942 #16kHz/30s
    width_time = 116
    num_filters = 128
    num_classes = 4
    model = Sequential()

    model.add(BatchNormalization(input_shape=(width_time, 128, 1)))
    model.add(Residual2D(filters=num_filters, diff_size=True))
    model.add(Residual2D(filters=num_filters, diff_size=False))
    model.add(Residual2D(filters=num_filters * 2, diff_size=True))
    model.add(Residual2D(filters=num_filters * 2, diff_size=False))
    model.add(Residual2D(filters=num_filters * 2, diff_size=False))
    model.add(Residual2D(filters=num_filters * 2, diff_size=False))
    model.add(Residual2D(filters=num_filters * 4, diff_size=True))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(units=num_filters * 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation='softmax'))
    optimizer = SGD(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


###############################################
## Model training #############################
## and ########################################
## testing ####################################
###############################################

# Trains and tests the model for a certain set of [epochs, batch_size, learning_rate],
# using cross-validation to get 100 different train/test folds to more accurately
# assess the model's performance.
def run_f1score(X, labels, epochs, batch_size, learning_rate, resume_training=False):
    print("começo a treinar:")
    # F1 Score with Repeated Stratified K Fold, 10 splits, 10 repeats, resulting
    # in 100 train/test folds
    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    acum, loss, scores_all = 0, [], []

    history_all_accuracy = []
    history_all_accuracy_val = []
    history_all_loss = []
    history_all_loss_val = []

    # Segment-level results (per clip == per segment)
    macro_f1_score_per_clip, f1_each_per_clip, confusion_matrix_global_per_clip = [], [], np.zeros((4, 4))

    # Sample-level results
    macro_f1_score, f1_each, confusion_matrix_global = [], [], np.zeros((4, 4))

    # save the model with the highest f1_score
    max_f1, max_f1_per_clip = 0, 0

    name_of_dir = 'bs_{}_ep_{}_lr_{}/'.format(batch_size, epochs, learning_rate)

    ###############################################
    ## Callbacks ##################################
    ###############################################
    th = MyCallbacks(threshold=0.9)
    ###############################################

    x_2_fold = np.array([1] * number_of_samples)
    y_2_fold = np.array([1] * number_of_samples)

    folds = list(kfold.split(x_2_fold, y_2_fold))
    folds_name = 'folds_' + str(batch_size) + '_' + str(epochs) + '_' + str(learning_rate)

    if resume_training:
        prev_data = load_prev_data(batch_size, epochs, learning_rate,
                                   directory='bs_{}_ep_{}_lr_{}'.format(batch_size, epochs, learning_rate))

        history_all_accuracy = prev_data[0]
        history_all_accuracy_val = prev_data[1]
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
        model = compile_model(learning_rate=learning_rate)

        ## Training data preparation for current fold##
        x_train, y_train = [], []
        for i in train:
            for temp in X[i]:
                x_train.append(temp)
                y_train.append(labels[i])

        x_train = np.asarray(x_train).reshape(len(x_train), window_width, 128, 1)
        y_train = np.asarray(y_train).reshape(len(y_train), 4)

        x_test, y_test = [], []
        x_test_chunk, y_test_chunk = [], []
        for j in test:
            for temp in X[j]:
                x_test_chunk.append(temp)
                y_test_chunk.append(labels[j])
            x_test.append(np.asarray(X[j]).reshape(len(X[j]), window_width, 128, 1))
            y_test.append(labels[j])

        x_test, y_test = np.asarray(x_test), np.asarray(y_test)
        x_test_chunk = np.asarray(x_test_chunk).reshape(len(x_test_chunk), width, 128, 1)
        y_test_chunk = np.asarray(y_test_chunk).reshape(len(y_test_chunk), 4)

        # train and save to log file
        history = model.fit(x=x_train, y=y_train,
                            validation_data=(x_test_chunk, y_test_chunk),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[th], verbose=1)

        print("\nLoss on last epoch:")
        print(history.history['loss'][-1])
        print("\nAcc on last epoch:")
        print(history.history['accuracy'][-1])
        loss.append(history.history['loss'][-1])

        # Save accuracy and loss from the training phase
        history_all_accuracy.append(history.history['accuracy'])
        history_all_accuracy_val.append(history.history['val_accuracy'])
        history_all_loss.append(history.history['loss'])
        history_all_loss_val.append(history.history['val_loss'])

        ## Testing portion ##
        print('Testing ' + str(acum) + '...')
        # Evaluate the model
        # Score is only calculated at segment-level
        score_temp = model.evaluate(x=x_test_chunk, y=y_test_chunk, verbose=0)
        print(score_temp)
        scores_all.append(score_temp)

        ## Segment-level results##
        # Predict test fold
        ypred_all = model.predict(x=x_test_chunk)

        # Calculate Confusion Matrix
        conf_temp_per_clip = confusion_matrix(np.asarray(y_test_chunk).argmax(axis=1),
                                              np.asarray(ypred_all).argmax(axis=1))
        print(conf_temp_per_clip)
        confusion_matrix_global_per_clip += conf_temp_per_clip

        # Predict for F1 Score
        f1_temp_per_clip = f1_score(np.asarray(y_test_chunk).argmax(axis=1), np.asarray(ypred_all).argmax(axis=1),
                                    average='macro')
        f1_each_per_clip.append(
            f1_score(np.asarray(y_test_chunk).argmax(axis=1), np.asarray(ypred_all).argmax(axis=1), average=None))
        print("\nF1_Score Segment-level: " + str(f1_temp_per_clip))
        # FIXME: CHANGE HERE
        if f1_temp_per_clip > max_f1_per_clip:
            max_f1_per_clip = f1_temp_per_clip
            best_acum_per_clip = acum
        macro_f1_score_per_clip.append(f1_temp_per_clip)

        ## Sample-level results ##
        y_pred_total = []
        y_mean_test_all = []

        # This loop predicts the quadrant for each segment pertaining to a sample,
        # averaging the quadrant as to get the overall sample prediction.
        # Other approaches use the mode instead of the average, but it was
        # found that the average performed better in our case.
        for test_sample in x_test:
            # Predict each chunk of the full song
            ypred = model.predict_on_batch(x=test_sample).reshape(len(test_sample), 4)
            # Save the corresponding quad of each chunk
            ypred_quad = np.asarray(ypred).argmax(axis=1)
            y_pred_total.append(ypred_quad)
            y_mean = np.around(np.mean(ypred_quad))
            y_mean_test_all.append(y_mean)

        conf_temp = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_mean_test_all))
        print(conf_temp)
        confusion_matrix_global += conf_temp

        f1_temp = f1_score(np.asarray(y_test).argmax(axis=1),
                           np.asarray(y_mean_test_all),
                           average='macro')
        f1_each.append(f1_score(np.asarray(y_test).argmax(axis=1),
                                np.asarray(y_mean_test_all),
                                average=None))

        if f1_temp > max_f1:
            max_f1 = f1_temp
        macro_f1_score.append(f1_temp)

        # Save metrics from the current test fold
        with open('history/' + name_of_dir + str(acum) + '.pickle', 'wb') as file_hist:
            hist_acum = History.History(history.history['accuracy'], history.history['val_accuracy'],
                                        history.history['loss'], history.history['val_loss'])
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
    print("\nSCORES:")
    print(scores_all)
    print("\nMACRO-F1-SCORE:")
    print(macro_f1_score)
    print("MEAN F1-SCORE:")
    print(avg(macro_f1_score))
    print("STD F1-SCORE:")
    print(np.asarray(macro_f1_score).std())
    print("MEAN LOSS:")
    print(avg(np.asarray(loss)))
    print("STD LOSS:")
    print(np.asarray(loss).std())
    print("MEAN F1-SCORE PER CLASS:")
    print("Q1\tQ2\tQ3\tQ4")
    print(str(avg(np.asarray(f1_each)[:, 0])) + '\t' + str(avg(np.asarray(f1_each)[:, 1])) + '\t' + str(
        avg(np.asarray(f1_each)[:, 2])) + '\t' + str(avg(np.asarray(f1_each)[:, 3])))
    print("CONFUSION MATRIX:")
    print(confusion_matrix_global)

    # save results
    # get the average f1 score
    idx = find_index_nearest(macro_f1_score, avg(macro_f1_score))
    hist = History.History(history_all_accuracy[idx],
                           history_all_accuracy_val[idx],
                           history_all_loss[idx],
                           history_all_loss_val[idx])

    return hist, macro_f1_score, f1_each, confusion_matrix_global, scores_all, \
           macro_f1_score_per_clip, f1_each_per_clip, confusion_matrix_global_per_clip


# Main function used for optimizing the model at hand.
def my_gridsearch_simple(x, Y):
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
        'batch_size': [50],  # removed 16
        'epochs': [100],
        'learning_rate': [0.001]  #
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
                    f1s_pc, f1e_pc, conf_pc = run_f1score(x, Y,
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
                    f1s_pc, f1e_pc, conf_pc = run_f1score(x, Y,
                                                          batch_size=matrix_of_params[0][bs],
                                                          epochs=matrix_of_params[1][ep],
                                                          learning_rate=matrix_of_params[2][lr])

                file_name = 'history_shortchunk_bs_' + str(matrix_of_params[0][bs]) + \
                            '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1s_name = 'f1_score_shortchunk_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1s_pc_name = 'f1_score_perclip_shortchunk_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1e_name = 'f1_each_shortchunk_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                f1e_pc_name = 'f1_each_perclip_shortchunk_bs_' + str(matrix_of_params[0][bs]) + \
                           '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                conf_name = 'conf_mat_shortchunk_bs' + str(matrix_of_params[0][bs]) + \
                            '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                conf_pc_name = 'conf_mat_perclip_shortchunk_bs' + str(matrix_of_params[0][bs]) + \
                            '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                scores_name = 'scores_shortchunk_bs' + str(matrix_of_params[0][bs]) + \
                              '_ep_' + str(matrix_of_params[1][ep]) + '_lr_' + str(matrix_of_params[2][lr])
                time_name = 'time_shortchunk_bs' + str(matrix_of_params[0][bs]) + \
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
                    time_file.write('Ran for {} minutes.'.format(time_curr / 60))

                time_elapsed = round(curr_grid_search - start_grid_search)
                print("GridSearch has run for {} minutes.".format(time_elapsed / 60))

                gridsearch_progress.next()

    end_grid_search = time.time()
    time_elapsed = round(end_grid_search - start_grid_search)
    print("GridSearch took {} minutes.".format(time_elapsed / 60))
    print("GridSearch complete.")
    return


def main(file_with_mel, file_with_labels):
    with open(file_with_mel + '.pickle', 'rb') as mel_file:
        df_spect = pickle.load(mel_file)
    with open(file_with_labels + '.pickle', 'rb') as sounds_file:
        sounds = pickle.load(sounds_file)

    _, _, songs_code = load_features_fixed()
    sounds = filterout(sounds, songs_code)
    labels = np.array([temp[2] for temp in sounds])

    # If samples were generated using the split_chunk_3_5s_to_mel function,
    # there is no need to reshape them.

    # Translate quadrants to ints
    mapping = {}
    classes = ['Q1', 'Q2', 'Q3', 'Q4']
    for x in range(len(classes)):
        mapping[classes[x]] = x
    Y_total = [mapping[temp] for temp in labels]

    Y_total = np_utils.to_categorical(Y_total)

    print("Preprocessing complete!")

    start = time.time()

    my_gridsearch_simple(df_spect, Y_total)

    end = time.time()
    elapsed = round(end - start)
    print("Minutes elapsed: " + str(elapsed / 60))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == "__main__":
    # Set seeds to more accurately replicate the reported results.
    random.seed(1)
    np.random.seed(1)

    file_with_samples = curr_path + ptd + samples_file
    file_with_mel = curr_path + ptd + mel_file

    # Set which device to use for computations
    # '' -> CPU; '0' -> GPU:0 ...
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Uncomment if you need to generate the samples and mel files
    # load2mel_fixed(True, True, file_with_samples)
    # create_melspectrograms(file_with_samples, file_with_mel)

    # Check your GPUs
    # print(get_available_gpus())

    main(file_with_mel, file_with_samples)

    print('Done')