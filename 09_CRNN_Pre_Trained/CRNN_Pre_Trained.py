import sys
sys.path.append('../Utils/')
import os
import pickle
import time
import datetime
from os import makedirs, listdir
from os.path import isfile, join
import random
import librosa
from pydub import AudioSegment
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import torchaudio
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path

from natsort import natsorted
import tqdm

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
samples_file = 'new_pub_complete_samples_normalized_16kHz'
# Name of the file containing all Mel-spectrogram representations of the samples.
# The program expects a pickle file with the structure: (num_of_samples, (width, height))
mel_file = 'new_pub_complete_dataset_16kHz_melspect_norm'
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
        sep=',', header=0)    # ordenar pelo song code (ascendente)
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


def take_song_code(item):
    return item[0]


def filterout(sounds, songs_code):
    # filtrar musicas sem features
    final_set = []
    for temp in sounds:
        # verificar se a musica tem features
        if temp[0] in songs_code:
            final_set.append(temp)
    # ordenar para ter a certeza que bate tudo certo
    final_set.sort(key=take_song_code)
    print("Retirei " + str(len(sounds) - len(final_set)) + " musicas!")
    return final_set


def get_melspect_from_torchaudio():
    with open(curr_path + ptd + samples_file,  'rb') as samp_file:
        samps = pickle.load(samp_file)

    melspect_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                               n_fft=512,
                                                               f_min=0.0,
                                                               f_max=8000.0,
                                                               n_mels=96)
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    melspect = []
    for samp in samps:
        the_tensor = torch.tensor(samp[1])
        melspect.extend(amp_to_db(melspect_transform(the_tensor)))


def load_prev_data(bs, curr_model):
    runs = natsorted([f for f in os.listdir('model_{}/bs_{}/'.format(curr_model, bs))
                      if isfile(join('model_{}/bs_{}/'.format(curr_model, bs), f))])

    f1_scores, f1_scores_each, conf_matrix_global = [], [], []

    for run in range(len(runs)):
        files_f1 = natsorted([f for f in os.listdir('model_{}/bs{}/run_{}'.format(curr_model, bs, run))
                           if isfile(join('model_{}/bs_{}/run_{}'.format(curr_model, bs, run), f))
                              and 'f1_score' in f and 'each' not in f])
        files_f1_each = natsorted([f for f in os.listdir('model_{}/bs{}/run_{}'.format(curr_model, bs, run))
                              if isfile(join('model_{}/bs_{}/run_{}'.format(curr_model, bs, run), f))
                              and 'f1_score_each' in f])
        files_conf = natsorted([f for f in os.listdir('model_{}/bs{}/run_{}'.format(curr_model, bs, run))
                                   if isfile(join('model_{}/bs_{}/run_{}'.format(curr_model, bs, run), f))
                                   and 'conf_mat' in f])

        t_f1_scores, t_f1_scores_each, t_conf_matrix_global = [], [], []
        for fi in files_f1:
            with open('model_{}/bs_{}/run_{}/{}'.format(curr_model, bs, run, fi), 'r') as open_file:
                t_f1_scores.append(pickle.load(open_file))
        for fi in files_f1_each:
            with open('model_{}/bs_{}/run_{}/{}'.format(curr_model, bs, run, fi), 'r') as open_file:
                t_f1_scores_each.append(pickle.load(open_file))
        for fi in files_conf:
            with open('model_{}/bs_{}/run_{}/{}'.format(curr_model, bs, run, fi), 'r') as open_file:
                t_conf_matrix_global.append(pickle.load(open_file))

        f1_scores.append(t_f1_scores)
        f1_scores_each.append(t_f1_scores_each)
        conf_matrix_global.append(t_conf_matrix_global)

    return [f1_scores, f1_scores_each, conf_matrix_global, len(runs)]


class MERDataset(data.Dataset):
    def __init__(self, split):
        self.samples = None
        self.targets = None
        self.targets_multiclass = None

        self.set_samples_targets(split)

    def set_samples_targets(self, split):
        file_with_samples = curr_path + ptd + samples_file
        file_with_mel = curr_path + ptd + mel_file

        with open(file_with_mel + '.pickle', 'rb') as mel_file:
            df_spect = pickle.load(mel_file)
        with open(file_with_samples + '.pickle', 'rb') as sounds_file:
            sounds = pickle.load(sounds_file)

        _, _, songs_code = load_features_fixed()
        sounds = filterout(sounds, songs_code)
        labels = np.array([temp[2] for temp in sounds])

        self.targets_multiclass = labels[split]

        mapping = {}
        classes = ['Q1', 'Q2', 'Q3', 'Q4']
        for x in range(len(classes)):
            mapping[classes[x]] = x
        Y_total = np.array([mapping[temp] for temp in labels])

        print("Preprocessamento completo!")

        self.samples = df_spect[split]
        self.targets = Y_total[split]

        print(self.samples.shape)
        print(self.targets.shape)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]


def prepare_datasets(train_split, test_split, bs):
    train_loader = data.DataLoader(dataset=MERDataset(train_split),
                                   batch_size=bs, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(dataset=MERDataset(test_split),
                                  batch_size=bs, shuffle=True, drop_last=False)

    return train_loader, test_loader

###############################################
## Model ######################################
###############################################
#########################################################################################
### CRNN model by Won et al. ################################################################
### https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py #############
#########################################################################################


# Custom Convolutional 2D layer,
class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class CRNN(nn.Module):
    '''
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    '''

    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96,
                 n_class=4):
        super(CRNN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, 64, pooling=(2, 2))
        self.layer2 = Conv_2d(64, 128, pooling=(3, 3))
        self.layer3 = Conv_2d(128, 128, pooling=(4, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(4, 4))

        # RNN
        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, n_class)

    def forward(self, x):
        # Spectrogram
        # This part is skipped to reduce the overhead at the start of each fold.
        # This can only be done due to the reduced size of the datasets used.
        # x = self.spec(x)
        # x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # RNN
        x = x.squeeze(2)
        # vvv Added vvv
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Softmax()(x)

        return x


# Solver class for trainining, optimization and evaluation.
# Repurposed from https://github.com/minzwon/sota-music-tagging-models/blob/master/training/solver.py
class Solver(object):
    def __init__(self, save_path, load_path, train_ds, test_ds, batch_size, cuda_not=False):
        # data loader
        self.train_dataloader = train_ds
        self.test_dataloader = test_ds

        # training settings
        self.model = None
        self.n_epochs = 200
        self.lr = 0.001
        self.optimizer = None
        self.use_tensorboard = True

        # model path and step size
        self.model_save_path = save_path
        self.model_load_path = load_path
        self.log_step = 10
        self.batch_size = batch_size
        self.model_type = 'crnn'

        self.f1_results = []
        self.f1_each_results = []
        self.conf_matrix = np.zeros((4, 4))

        self.f1_results_pc = []
        self.f1_each_results_pc = []
        self.conf_matrix_pc = []

        # cuda
        if not cuda_not:
            self.is_cuda = torch.cuda.is_available()
        else:
            self.is_cuda = False

    def get_results_pc(self):
        return self.f1_results

    def get_each_pc(self):
        return self.f1_each_results

    def get_confused_pc(self):
        return self.conf_matrix

    def build_model(self):
        # model
        model = CRNN()

        # cuda
        if self.is_cuda:
            model.cuda()

        # Model has to be returned, or else it is not set
        return model

    # Load pre-trained model from file. Can be a model trained from either:
    # -> MagnaTagATune; -> MGT-Jamendo; -> MSD (provided split)
    def load_prep_pretrained_model(self):
        # load pretrained model
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)

            total_model_params = [i for i, param in enumerate(self.model.parameters())][-1]

            for count, param in enumerate(self.model.parameters()):
                if count < total_model_params - 1:
                    param.requires_grad = False
                else:
                    pass
            print()
            print('Loaded model')
            time.sleep(2)

    # Loads full model from file, pops layers not relevant for this approach.
    def load(self, filename):
        S = torch.load(filename)
        # if 'spec.mel_scale.fb' in S.keys():
        # self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        S.pop('spec.spectrogram.window')
        S.pop('spec.mel_scale.fb')
        S.pop('dense.weight')
        S.pop('dense.bias')
        self.model.load_state_dict(S, strict=False)

    # Loads full model from file.
    def load_local(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S, strict=False)

    # In PyTorch, a sample can only be sent to a GPU after
    # being cast as a PyTorch Variable.
    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    # Mimics the information given at each epoch by Tensorflow
    def print_log(self, epoch, ctr, loss, start_t):
        if (ctr) % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                  (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   epoch + 1, self.n_epochs, ctr, len(self.train_dataloader), loss.item(),
                   datetime.timedelta(seconds=time.time() - start_t)))

    # Training and optimizer loop
    def train(self):
        # Start training
        start_t = time.time()
        reconst_loss = nn.CrossEntropyLoss()
        best_metric = 0
        drop_counter = 0
        acum = 0

        self.model = self.build_model()
        self.load_prep_pretrained_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)
        current_optimizer = 'adam'

        for epoch in range(self.n_epochs):
            print('Epoch {}'.format(epoch))
            self.model = self.model.train()
            ctr = 0
            drop_counter += 1

            for x, y in tqdm.tqdm(self.train_dataloader):
                ctr += 1
                # Forward
                x = self.to_var(torch.tensor(x))
                y = self.to_var(torch.tensor(y, dtype=torch.long))
                out = self.model(x)

                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)

            # validation
            best_metric = self.validation(best_metric, epoch, acum)

            # schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
              % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 datetime.timedelta(seconds=time.time() - start_t)))

    # Optimizer scheduler.
    # The base implementation uses this scheduler to converge faster to a solution,
    # should be within 200 epochs. This was originally used for the MTG-Jamendo datset,
    # which is considerably larger than our own datasets.
    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 80:
            if os.path.isfile('best_model_{}.pth'.format(self.batch_size)):
                self.load_local('best_model_{}.pth'.format(self.batch_size))
                self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001,
                                                 momentum=0.9, weight_decay=0.0001,
                                                 nesterov=True)
                current_optimizer = 'sgd_1'
                drop_counter = 0
                print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            if os.path.isfile('best_model_{}.pth'.format(self.batch_size)):
                self.load_local('best_model_{}.pth'.format(self.batch_size))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.0001
                current_optimizer = 'sgd_2'
                drop_counter = 0
                print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            if os.path.isfile('best_model_{}.pth'.format(self.batch_size)):
                self.load_local('best_model_{}.pth'.format(self.batch_size))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.00001
                current_optimizer = 'sgd_3'
                print('sgd 1e-5')
        return current_optimizer, drop_counter

    # Evaluates the trained model for the current fold, writing the results to the respective files.
    def validation(self, best_metric, epoch):
        f1_score_macro, f1_score_each, conf_matrix, loss = self.get_validation_score(epoch)

        self.f1_results.append(f1_score_macro)
        self.f1_each_results.append(f1_score_each)
        self.conf_matrix += conf_matrix

        with open(os.path.join(self.model_save_path, 'loss_{}_bs_{}'.format(epoch + 1, self.batch_size) + '.pickle'),
            'wb') as loss_file:
            pickle.dump(loss, loss_file)
        with open(os.path.join(self.model_save_path, 'f1_score_{}_bs_{}'.format(epoch + 1, self.batch_size) + '.pickle'),
            'wb') as f1_file:
            pickle.dump(f1_score_macro, f1_file)
        with open(os.path.join(self.model_save_path, 'f1_score_each_{}_bs_{}'.format(epoch + 1, self.batch_size) + '.pickle'),
            'wb') as f1_each_file:
            pickle.dump(f1_score_each, f1_each_file)
        with open(os.path.join(self.model_save_path, 'conf_mat_{}_bs_{}'.format(epoch + 1, self.batch_size) + '.pickle'),
            'wb') as conf_file:
            pickle.dump(conf_matrix, conf_file)

        score = 1 - loss
        if score > best_metric:
            print('best model!')
            best_metric = score
            torch.save(self.model.state_dict(),
                       'best_model_{}.pth'.format(self.batch_size))
        return best_metric

    # Helper function for predicting the labels using the model and calculates the relevant metrics.
    def get_validation_score(self, epoch):
        self.model = self.model.eval()
        est_array, est_array_full = [], []
        gt_array = []
        losses = []
        reconst_loss = nn.CrossEntropyLoss()
        index = 0
        for x, y in tqdm.tqdm(self.test_dataloader):
            # forward
            x = self.to_var(torch.tensor(x))
            y = self.to_var(torch.tensor(y, dtype=torch.long))
            out = self.model(x)

            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            est_array.extend(np.argmax(out.numpy(), axis=1))
            gt_array.extend(y.cpu().numpy())

            index += 1

        est_array, gt_array = \
            np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print('loss: %.4f' % loss)

        f1_score_macro = metrics.f1_score(gt_array, est_array, average='macro')
        f1_score_each = metrics.f1_score(gt_array, est_array, average=None)
        conf_matrix = metrics.confusion_matrix(gt_array, est_array)

        return f1_score_macro, f1_score_each, conf_matrix, loss


def run_for_folds(folds):
    batch_size = [16]
    model = ['mtat', 'jam', 'msd']

    resume_params = [
    ]

    ran_params = [
    ]

    for curr_model in model:
        for bs in batch_size:
            if [bs, curr_model] in ran_params:
                print('Ran for {} with {} batch size. Proceeding...'.format(curr_model, bs))

            f1_scores = []
            f1_scores_each = []
            conf_matrix_global = np.zeros((4, 4))
            count = 0

            if [bs, curr_model] in resume_params:
                prev_data = load_prev_data(bs, curr_model)

            time_start = time.time()

            for train, test in folds:
                count += 1

                tr, ts = prepare_datasets(train, test, bs)
                dir_path = 'model_{}/bs_{}/run_{}/'.format(curr_model, bs, count)

                try:
                    os.makedirs(dir_path)
                except FileExistsError:
                    pass

                to_solve = Solver(dir_path, 'pretrained_models/best_model_' + curr_model + '.pth', tr, ts, bs)
                to_solve.train()

                f1_scores.append(to_solve.get_results())
                f1_scores_each.append(to_solve.get_each())
                conf_matrix_global += to_solve.get_confused()
            time_end = time.time()

            f1_scores = np.mean(np.array(f1_scores), axis=1)
            f1_scores_each = np.mean(np.array(f1_scores_each), axis=1)

            with open('final_results_f1_model_{}_bs_{}.txt'.format(curr_model, bs), 'w') as fr:
                print('Final Score is: {}'.format(f1_scores.mean()))
                fr.write('Final F1 Score Macro:\n')
                fr.write(str(f1_scores.mean()) + '\n')
                print('Final Score Each: {}'.format(f1_scores_each))
                fr.write('Final F1 Each:\n')
                fr.write(str(f1_scores_each) + '\n')
                print('Final Confusion Matrix:')
                print(str(conf_matrix_global))
                fr.write('Total time: {}'.format(round(time_end - time_start) / 60))

    print('Finished!')


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    x_2_fold = np.array([1] * number_of_samples)
    y_2_fold = np.array([1] * number_of_samples)

    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    folds = list(kfold.split(x_2_fold, y_2_fold))

    # TODO: test to understand wrong number of results per fold
    run_for_folds(folds)

