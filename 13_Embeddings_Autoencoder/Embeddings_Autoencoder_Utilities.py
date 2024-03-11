import os
import sys
from pathlib import Path
sys.path.append('../Utils/')
import pickle
import time
import random
import numpy as np
from os import makedirs
from keras.utils import np_utils
from tensorflow.keras.layers import Dense, Input, InputLayer, Embedding, Reshape, LeakyReLU, Conv2DTranspose, \
    concatenate, Lambda, \
    BatchNormalization,MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

from tqdm import tqdm
from History import History

###################
## Options #########
##################
curr_path = Path('.').absolute().parent.__str__()
# Path to the directory with all necessary data
ptd = 'New_MERGE_Complete_Data/'
# Location of the .wav files converted from .mp3 as the intermediate step to generate Mel-spectrograms
location = 'total_wav/'
# Name of the file containing all samples. The program expects a pickle file
# with the structure: (num_samples, (song_id, waveform, target))
samples_file = 'new_pub_complete_samples_normalized_22kHz'
# Name of the file containing all Mel-spectrogram representations of the samples.
# The program expects a pickle file with the structure: (num_of_samples, (width, height))
mel_file = 'new_pub_complete_dataset_22kHz_melspect_norm'

embeddings_file = 'embedded_samples_16kHz_norm_npc'
hist_file = 'hist_ac'
weights_name = 'autoencoder_weights'

width_time = 942
epochs = 150
batch_size = 150

####################
## Utils ##############
####################
def normalize_11(sample):
    average = (sample.min() + sample.max()) / 2
    range_sample = (sample.max() - sample.min()) / 2
    normalized_x = (sample - average) / range_sample
    return normalized_x, average, range_sample

def load_scale_data(data):
    new_data = []
    avg, avg_range = [], []
    for sample in data:
        # new_data.append(2*((sample - sample.min()) / (sample.max() - sample.min())) + (-1))
        norm_sample, average_sample, range_sample = normalize_11(sample)
        new_data.append(norm_sample)
        avg.append(average_sample)
        avg_range.append(range_sample)
    return np.asarray(new_data), np.average(avg), np.average(avg_range)


###########
# Models ##
###########
######################################################
## Autoencoder model -> Bottleneck dimension is 60416; the  ####
## max reduction possible before being unable to reconstruct ###
## the input samples in an acceptable quality. ###############
#####################################################
def autoEncoder(optimizer='adam'):
    # ENCODER
    input_img = Input((width_time, 128, 1))
    # downsample till dense layers
    encoder = Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', name='disc_2')(input_img)
    encoder = LeakyReLU(alpha=0.2)(encoder)
    encoder = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='disc_3')(encoder)
    encoder = LeakyReLU(alpha=0.2)(encoder)
    encoder = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', name='disc_4')(encoder)
    encoder = LeakyReLU(alpha=0.2)(encoder)
    encoder = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same', name='disc_5')(encoder)
    encoder = LeakyReLU(alpha=0.2)(encoder)

    # MID
    middle = Flatten()(encoder)
    smiddle = Reshape((59, 8, 128))(middle)

    # DECODER
    decoder = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', name='gen_6')(smiddle)
    decoder = LeakyReLU(alpha=0.2)(decoder)
    decoder = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', name='gen_1')(decoder)
    decoder = LeakyReLU(alpha=0.2)(decoder)
    # upsample to 32x246
    decoder = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', name='gen_2')(decoder)
    decoder = LeakyReLU(alpha=0.2)(decoder)
    # upsample to 64x492
    decoder = Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', name='gen_3')(decoder)
    decoder = LeakyReLU(alpha=0.2)(decoder)
    # upsample to 128x984
    # output
    decoder = Conv2D(1, (5, 5), activation='tanh', padding='same', name='gen_5')(decoder)
    decoder = Lambda(lambda x: x[:, :942, :, :])(decoder)
    model = Model(input_img, decoder)
    opt = Adam(learning_rate=0.0002)
    model.compile(loss='mse', optimizer=opt, metrics=[RootMeanSquaredError()])
    print(model.summary())
    return model

####################
## Model training #####
####################
# Train autoencoder and save weights
def train_autoencoder(x, epochs, batch_size):
    model = autoEncoder()

    hist = model.fit(x, x,
                     epochs=epochs, batch_size=batch_size)

    with open(hist_file + '.pickle', 'wb') as ac_hist:
        pickle.dump(History(None, None, hist.history['loss'], None), ac_hist)
    model.save_weights(weights_name + '.h5')

    print('Done training autoencoder')


# Generate embeddings == get latent space representations (in this case)
def latent_space_from_encoder(x, labels, filepath):
    # separate classes
    q1_i = []
    q2_i = []
    q3_i = []
    q4_i = []
    acum = 0
    for y_temp in labels:
        if y_temp == 'Q1':
            q1_i.append(acum)
        if y_temp == 'Q2':
            q2_i.append(acum)
        if y_temp == 'Q3':
            q3_i.append(acum)
        if y_temp == 'Q4':
            q4_i.append(acum)
        acum = acum + 1
    x_i = [q1_i, q2_i, q3_i, q4_i]

    # assign model
    model = autoEncoder()
    model.load_weights(filepath, by_name=True)

    encoder_input = Input(shape=(width_time, 128, 1))
    model.layers[1].trainable = False
    encoder_output = model.layers[1](encoder_input)
    for l in model.layers[2:10]:
        l.trainable = False
        encoder_output = l(encoder_output)
    encoder_model = Model(inputs=encoder_input, outputs=encoder_output)
    opt = Adam(learning_rate=0.0002)
    encoder_model.compile(loss='mse', optimizer=opt, metrics=[RootMeanSquaredError()])
    encoder_model.summary()

    # predict on X to get distribution per class
    quad_number = 0
    for quad in tqdm(x_i):
        y_temp = encoder_model.predict(x[quad])
        with open(curr_path + ptd + embeddings_file + '.pickle', 'wb') as emb_file:
            pickle.dump(y_temp, emb_file)
        quad_number = quad_number + 1
    return


def main_train(file_with_mel, file_with_labels):
    with open(file_with_mel + '.pickle', 'rb') as mel_file:
        df_spect = pickle.load(mel_file)
    with open(file_with_labels + '.pickle', 'rb') as samp_file:
        sounds = pickle.load(samp_file)
    sounds.sort(key=lambda sounds: sounds[0])

    X = df_spect.reshape(len(df_spect), width_time, 128, 1)
    X_norm, avg, avg_range = load_scale_data(X)

    labels = [sound[2] for sound in sounds]
    # converter para inteiros
    mapping = {}
    classes = ['Q1', 'Q2', 'Q3', 'Q4']
    for x in range(len(classes)):
        mapping[classes[x]] = x
    Y_total = [mapping[temp] for temp in labels]

    Y_total = np_utils.to_categorical(Y_total)

    print(X_norm.shape)

    print("Preprocessamento completo!")

    start = time.time()

    train_autoencoder(X_norm, epochs, batch_size)
    latent_space_from_encoder(X_norm, labels, weights_name + '.h5')

    end = time.time()
    elapsed = round(end - start)
    print("Minutos passados: " + str(elapsed / 60))


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)

    # Set which device to use for computations
    # '' -> CPU; '0' -> GPU:0
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Train autoencoder, save weigths, and get latent
    # space representation of samples for each quadrant.
    main_train(mel_file, samples_file)

    print('Done')

