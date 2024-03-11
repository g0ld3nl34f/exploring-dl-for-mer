import os
from os import listdir
from os.path import join, isfile
import pickle
import random
import time

import numpy as np
from tensorflow.keras.layers import Dense, Input, InputLayer, Embedding, Reshape, LeakyReLU, Conv2DTranspose, \
    concatenate, Lambda, \
    BatchNormalization,MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from keras.utils import np_utils
import smote_variants as sv
from natsort import natsorted
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tqdm import tqdm

from History import History
from sklearn.metrics.pairwise import pairwise_distances


def normalize_11(sample):
    average = (sample.min() + sample.max()) / 2
    range_sample = (sample.max() - sample.min()) / 2
    normalized_x = (sample - average) / range_sample
    return normalized_x, average, range_sample


def denormalize(sample, avg, avg_range):
    return (sample * avg_range) + avg

def load_scale_data(data):
    new_data = []
    avg, avg_range = [], []
    for sample in data:
        # new_data.append(2*((sample - sample.min()) / (sample.max() - sample.min())) + (-1))
        norm_sample, average_sample, range_sample = normalize_11(sample)
        new_data.append(norm_sample)
        avg.append(average_sample)
        avg_range.append(range_sample)
        # sns.displot(np.asarray(new_data).flatten())
        # sns.displot(np.asarray(normalize_11(sample)).flatten())
        # plt.show()
    return np.asarray(new_data), np.average(avg), np.average(avg_range)

class AutoEmbedder(tf.keras.models.Model):
    def __init__(self, lr=0.0002):
        super(AutoEmbedder, self).__init__()
        self.kernel_size = (5, 5)
        self.latent_dim = 1000
        self.opt = Adam(lr)

        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()

    def init_encoder(self):
        encoder = Sequential([
            Input(shape=(942, 128, 1)),

            Conv2D(filters=16, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='disc_1'),
            LeakyReLU(alpha=0.2, name='leaky_relu_emb_1'),

            Conv2D(filters=32, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='disc_2'),
            LeakyReLU(alpha=0.2, name='leaky_relu_emb_2'),

            Conv2D(filters=64, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='disc_3'),
            LeakyReLU(alpha=0.2, name='leaky_relu_emb_3'),

            Conv2D(filters=128, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='disc_4'),
            LeakyReLU(alpha=0.2, name='leaky_relu_emb_4'),

            Flatten(name='flat')
        ])

        return encoder

    def init_decoder(self):
        decoder = Sequential([
            Reshape((59, 8, 128), input_shape=(60416,)),

            Conv2DTranspose(128, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='gen_1'),
            LeakyReLU(alpha=0.2, name='leaky_relu_demb_1'),

            Conv2DTranspose(64, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='gen_2'),
            LeakyReLU(alpha=0.2, name='leaky_relu_demb_2'),

            Conv2DTranspose(32, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='gen_3'),
            LeakyReLU(alpha=0.2, name='leaky_relu_demb_3'),

            Conv2DTranspose(16, kernel_size=self.kernel_size, strides=(2, 2), padding='same', name='gen_4'),
            LeakyReLU(alpha=0.2, name='leaky_relu_demb_4'),

            Conv2D(1, (5, 5), activation='tanh', padding='same'),
            Lambda(lambda x: x[:, :942, :, :], name='out_decoder')
        ])

        return decoder

    def models_summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def call(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def save_weights_of_autoembedder(self, dir_name, name_of_model):
        self.encoder.save_weights(dir_name + 'encoder_{}.h5'.format(name_of_model))
        self.decoder.save_weights(dir_name + 'decoder_{}.h5'.format(name_of_model))

    def load_weights_of_autoembedder(self, dir_name, name_of_model):
        self.encoder.load_weights(dir_name + 'encoder_{}.h5'.format(name_of_model), by_name=True)
        self.decoder.load_weights(dir_name + 'decoder_{}.h5'.format(name_of_model), by_name=True)


def train_model(bs, ep, lr=0.0002):
    dir_name = 'trained_models/'

    try:
        os.makedirs(dir_name)
    except FileExistsError:
        pass

    with open(
            '/home/pedrolouro/experiments-msc/New_Public_DB_Code/CNN_Simple_Sa/new_pub_complete_dataset_16kHz_melspect_norm.pickle',
            'rb') as orig_file:
        originals = pickle.load(orig_file)
        originals = originals.reshape(originals.shape[0], originals.shape[1], originals.shape[2], 1)

    originals_norm, avg, avg_range = load_scale_data(originals)
    #test_denormalized = denormalize_11(originals_norm[123], avg, avg_range)
    #bef_test = originals[123]

    start_time = time.time()

    #for train, test in folds:
    model = AutoEmbedder(lr)
    model.compile(loss='mse', optimizer=model.opt, metrics=[RootMeanSquaredError()])
    model.models_summary()

    hist = model.fit(x=originals_norm, y=originals_norm,
                     validation_data=[originals_norm, originals_norm],
                       epochs=ep, batch_size=bs)

    #if acum == 1:
        #model.summary()

    the_history = History(None, None, hist.history['loss'], None)

    with open(dir_name + 'history_new_pub_comp_ae.pickle', 'wb') as blocks_7:
        pickle.dump(the_history, blocks_7)
    model.save_weights_of_autoembedder(dir_name, 'best_new_pub_comp_ae_adam')

    print('Done training autoencoder.')


def get_embeddings(file_with_mel, file_with_samples):
    width_time = 942
    batch_size = 32
    # file_with_mel = '/home/pedrolouro/experiments-msc/Updated_Code/CNN_Voice_Separated/dataset_16kHz_melspect_norm_fixed'
    # Experiment with 96 mel bands

    with open(file_with_mel + '.pickle', 'rb') as mel_file:
        df_spect = pickle.load(mel_file)
    with open(file_with_samples + '.pickle', 'rb') as sounds_file:
        sounds = pickle.load(sounds_file)

    df_spect = df_spect.reshape(len(df_spect), 942, 128, 1)

    embed_model = AutoEmbedder()
    embed_model.compile(loss='mse', optimizer=embed_model.opt, metrics=[RootMeanSquaredError()])
    embed_model.load_weights_of_autoembedder('trained_models/', 'best_new_pub_comp_ae_adam')

    X_normalized, avg, avg_range = load_scale_data(df_spect)

    embedded_samples = []

    for i in tqdm(range(0, X_normalized.shape[0], batch_size)):
        if i < X_normalized.shape[0]:
            embedded_samples.extend(embed_model.encoder.predict_on_batch(X_normalized[i: i + batch_size]))

    with open('embedded_samples_16kHz_norm_npc_' + '.pickle', 'wb') as emb_file:
        pickle.dump(embedded_samples, emb_file)


def get_mapping(quad):
    mapping = {}
    classes = ['Q1', 'Q2', 'Q3', 'Q4']
    for x in range(len(classes)):
        mapping[classes[x]] = x
    return np_utils.to_categorical(mapping[quad], num_classes=len(classes))


def smote_class_training_set(x, y, num_new_samples, class_to_balance):
    class_to_balance = np.argmax(get_mapping(class_to_balance))

    x = np.array(x)
    y = np.argmax(y, axis=1)

    num_samples_quad_to_smote = [count for count, i in enumerate(y) if i == class_to_balance]

    for quad in range(4):
        if quad != class_to_balance:
            class_samples = [count for count, i in enumerate(y) if i == quad]
            dif_samples = len(class_samples) - len(num_samples_quad_to_smote)

            if num_new_samples - dif_samples > 0:
                choices = np.random.choice(class_samples, num_new_samples - dif_samples, replace=False)
                x = np.append(x, x[choices], axis=0)
                y = np.append(y, y[choices], axis=0)

            print('Class {}:'.format(quad))
            print(len([i for i in y if i == quad]))

    smote_sampler = sv.Borderline_SMOTE2(proportion=2.0, random_state=1)
    oversampler = sv.MulticlassOversampling(oversampler=smote_sampler,
                                            strategy='equalize_1_vs_many')

    x_smoted, y_smoted = oversampler.sample(x, y)

    print('Size before smote: x {},\t after: x_smote {}'.format(len(x), len(x_smoted)))
    print('\t y {},\t y_smote {}'.format(len(x), len(y_smoted)))

    x_smoted, y_smoted = x_smoted[-num_new_samples:], y_smoted[-num_new_samples:]
    return x_smoted, y_smoted


def generate_samples_from_embs(file_with_embs, file_with_samples, num_to_gen, variant='SMOTE'):
    with open(file_with_embs + '.pickle', 'rb') as embs_file:
        embs = pickle.load(embs_file)
    with open(file_with_samples + '.pickle', 'rb') as sounds_file:
        sounds = pickle.load(sounds_file)

    sounds.sort(key=lambda sounds: sounds[0])

    labels = np.array([temp[2] for temp in sounds])

    # converter para inteiros
    mapping = {}
    classes = ['Q1', 'Q2', 'Q3', 'Q4']
    for x in range(len(classes)):
        mapping[classes[x]] = x
    Y_total = [mapping[temp] for temp in labels]

    Y_total = np_utils.to_categorical(Y_total)

    print(Y_total[0])
    print("Preprocessamento completo!")

    start = time.time()

    for cl in classes:
        print('Generating samples for class {}...'.format(cl))
        #FIXME: Class or binary smote
        #x_smoted, y_smoted = smote_raw_training_set(embs, Y_total, num_to_gen, cl)
        x_smoted, y_smoted = smote_class_training_set(embs, Y_total, num_to_gen, cl)

        samps_to_save = []
        for samp_and_lab in zip(x_smoted, y_smoted):
            samps_to_save.append([samp_and_lab[0], samp_and_lab[1]])

        with open('embedded_smote/gen_samples_embs_{}_{}_{}.pickle'.format(variant, cl, num_to_gen), 'wb') as gen_file:
            pickle.dump(samps_to_save, gen_file)

    end = time.time()

    print('Took {} seconds to generate all samples.'.format(round(end - start)))


def view_mel_and_mel_after_emb():
    with open(
            '/home/pedrolouro/experiments-msc/Updated_Code/CNN_Voice_Separated/dataset_16kHz_melspect_norm_fixed.pickle',
            'rb') as orig_file:
        originals = pickle.load(orig_file)
    with open('autoencoder_results.pickle', 'rb') as retrieved_file:
        retrieved_from_embs = pickle.load(retrieved_file)

    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(np.rot90(originals[0]),
                  cmap=pyplot.get_cmap('inferno'))
    axs[1].imshow(np.rot90(retrieved_from_embs[0]),
                  cmap=pyplot.get_cmap('inferno'))
    pyplot.show()

def mapping_quads(target):
    if target==0:
        return 'Q1'
    elif target==1:
        return 'Q2'
    elif target==2:
        return 'Q3'
    elif target==3:
        return 'Q4'
    else:
        print('How did you get here, mate?')
        print('Targets should be between 0 and 3.')
        raise ValueError


def decode_smoted_embeddings(emb_folder,
                             deemb_folder='decoded_smote/',
                             variant_used='Borderline_SMOTE2',
                             unique_num_samples=4,
                             model_folder='trained_models/',
                             model_name='best_original_arch_adam',
                             batch_size=32):
    ##
    # SMOTE samples come in the shape -> (samples, (embedded_sample, target))
    ##
    files_with_smoted_embs = natsorted(list({f for f in listdir(emb_folder) if isfile(join(emb_folder, f))}))
    all_embeddings = []

    for i in range(0, len(files_with_smoted_embs), unique_num_samples):
        if i+unique_num_samples <= len(files_with_smoted_embs):
            this_class = []
            for file_e in files_with_smoted_embs[i:i+unique_num_samples]:
                with open(emb_folder + file_e, 'rb') as curr_file:
                    this_class.append(pickle.load(curr_file))
            all_embeddings.append(this_class)

    model = AutoEmbedder()
    model.compile(loss='mse', optimizer=model.opt, metrics=[RootMeanSquaredError()])
    model.load_weights_of_autoembedder(model_folder, model_name)

    try:
        os.makedirs(deemb_folder)
    except FileExistsError:
        pass

    for count, quad_embeds in enumerate(all_embeddings):
        for curr_embs in quad_embeds:
            embs_samples = np.array([samp[0] for samp in curr_embs])
            curr_decodings = []
            for i in tqdm(range(0, len(embs_samples), batch_size)):
                if i < len(embs_samples) and i + batch_size <= len(embs_samples):
                    curr_decodings.extend(model.decoder.predict_on_batch(embs_samples[i:i+batch_size]))
                elif i < len(embs_samples) < i+batch_size:
                    curr_decodings.extend(model.decoder.predict_on_batch(embs_samples[i:]))
                    break
            with open(deemb_folder + 'gen_samples_{}_{}_{}.pickle'.format(
                        variant_used, mapping_quads(count), len(curr_decodings)), 'wb') as to_save:
                pickle.dump(np.array(curr_decodings), to_save)

    print('All embedded samples decoded and saved!')


def compare_images():
    with open('/home/pedrolouro/experiments-msc/Updated_Code/CNN_Voice_Separated/dataset_16kHz_melspect_norm_fixed.pickle', 'rb') as orig_file:
        originals = pickle.load(orig_file)

    originals = originals.reshape(originals.shape[0], originals.shape[1], originals.shape[2], 1)
    chosen = np.random.choice(len(originals), 1)[0]

    model = AutoEmbedder()
    model.encoder.load_weights('trained_models/encoder_best_bs_150_ep_100_lr_0.0002.h5')
    model.decoder.load_weights('trained_models/decoder_best_bs_150_ep_100_lr_0.0002.h5')

    reconstructed = model.call(originals[chosen].reshape(1, 942, 128, 1))
    reconstructed = reconstructed.numpy()[0]
    the_orig = originals[chosen]

    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(np.rot90(originals[chosen]),
                  cmap='inferno')
    axs[1].imshow(np.rot90(reconstructed),
                  cmap='inferno')
    pyplot.show()

def load_all_encoded_samples(dir_path, variant, quad, num_samples):
    with open(dir_path +
              'gen_samples_embs_' + variant + '_' +
              quad + '_' + num_samples + '.pickle', 'rb') as sample_file:
        these_samples = pickle.load(sample_file)

    return [samp[0] for samp in these_samples]


def similarity_measure_bt_gen_samps(dir_path, variant, num_samples, targets):
    ##
    # Samples on decoded_smote folder: (num_gen_samples, width, height, channels)

    all_samples = []
    for c in targets:
        all_samples.append(
            load_all_encoded_samples(dir_path, variant, c, str(num_samples)))

    for i in range(len(targets)):
        print('Similarity between samples in {}: {}'.format(
            targets[i], pairwise_distances(all_samples[i],
                                           n_jobs=1)
        ))

def see_melspect(to_show):
    with open(to_show + '.pickle', 'rb') as ff:
        some = pickle.load(ff)

    fig, axs = pyplot.subplots(1, 2)
    axs[0].imshow(np.rot90(some[23]),
                  cmap='inferno')
    axs[1].imshow(np.rot90(some[41]),
                  cmap='inferno')
    pyplot.show()

if __name__ == '__main__':
    ##
    # Steps to reproduce:
    #   - Train model, if not available, with 4 conv blocks (16, 32, 64, 128)
    #       (Make sure to normalize the melspecs and use Adam with lr=0.0002)
    #   - Embedd all samples using the encoder part of the train model
    #   - Apply desired SMOTE variation
    #   - Use decoder to retrieve image
    #   - (Optional) Denormalize images for consistent values between all samples

    random.seed(1)
    np.random.seed(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # create_embeddings()
    # create_deembedding()

    fwe = 'embedded_samples_16kHz_norm_npc_'
    fwm = '/home/pedrolouro/experiments-msc/New_Public_DB_Code/CNN_Simple_Sa/new_pub_complete_dataset_16kHz_melspect_norm'
    fwl = '/home/pedrolouro/experiments-msc/New_Public_DB_Code/CNN_Simple_Sa/new_pub_complete_samples_normalized_16kHz'
    embedded_folder = 'embedded_smote/'

    epochs = 150
    batch_size = 150

    #train_model(batch_size, epochs, lr=0.0002)
    #get_embeddings(fwm, fwl)

    #for i in [25]: # 50, 100, 150
        #generate_samples_from_embs(fwe, fwl, i, 'Borderline_SMOTE2')
        #similarity_measure_bt_gen_samps('embedded_smote/',
                                        #'Borderline_SMOTE2', i,
                                        #['Q1', 'Q2', 'Q3', 'Q4'])

    decode_smoted_embeddings(embedded_folder, unique_num_samples=1, variant_used='Borderline_SMOTE2',
                             model_name='best_new_pub_comp_ae_adam')
    #see_melspect('decoded_smote/gen_samples_Borderline_SMOTE2_Q3_100')
