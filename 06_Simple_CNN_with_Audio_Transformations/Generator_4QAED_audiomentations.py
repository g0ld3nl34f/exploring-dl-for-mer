import pickle
import librosa
import keras.utils.data_utils
import numpy as np
from progress.bar import IncrementalBar
from matplotlib import pyplot


class Generator_4QAED(keras.utils.data_utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.samples = x
        self.targets = y
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(
            len(self.targets) / float(self.batch_size))
        ).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []

        for samp, target in zip(batch_x, batch_y):
            X.append(samp)
            y.append(target)

        return np.asarray(X), np.asarray(y)


def separate_melspectrograms(file_with_samples, directory='.', width_time=942):
    # Width time depends on sample rate, 942 == 30s/16kHz sample rate
    with open(file_with_samples + ".pickle", "rb") as sounds_file:
        sounds = pickle.load(sounds_file)

    print("Loading dataset...")
    spect_all = [librosa.power_to_db(librosa.feature.melspectrogram(temp[1])) for temp in sounds]
    print("Dataset loaded successfully!")

    # See melspect before rotation
    pyplot.imshow(spect_all[0], cmap=pyplot.get_cmap('inferno'))

    _ = input('Melspect before rotation.\nEnter to continue')

    # See melspect after rotation
    pyplot.imshow(np.rot90(spect_all[0]))

    _ = input('Melspect after rotation.\nEnter to continue')

    spect_all_rotated = [np.rot90(np.pad(temp, ((0, 0), (0, width_time - len(temp[0]))))) for temp in spect_all]

    pyplot.imshow(spect_all_rotated[0])

    _ = input('Melspect after rotation and padding.\nEnter to continue')

    mel_process = IncrementalBar('Turning into mel...', max=len(spect_all_rotated))
    for i, spect in enumerate(spect_all_rotated):
        curr_spect = [sounds[i][0], spect, sounds[i][2]]

        with open(directory + '/song_melspectrogram_' + str(i) + '.pickle', 'wb') as file_with_mel:
            pickle.dump(curr_spect, file_with_mel)

        mel_process.next()

    print("Process complete!")

