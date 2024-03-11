import numpy as np
import librosa
import random


class Augmented:

    def __init__(self):
        self.time_shift_options = ['right', 'left', 'both']
        self.pitch_shift_options = ['up', 'down']
        self.time_stretch_options = ['faster', 'slower']

    def time_shift(self, data, sampling_rate, shift_max=5):
        # default _> shift_max 3 seconds
        shift = np.random.randint(sampling_rate * shift_max)
        shift_direction = self.time_shift_options[round(random.uniform(0, 2))]
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        # Set to silence for the beginning or the end
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    def pitch_shift(self, data, sampling_rate, pitch_factor=2):
        shift_direction = self.time_shift_options[round(random.uniform(0, 1))]
        if shift_direction == 'down':
            pitch_factor = -pitch_factor
        # default _> pitch 1 tone (2 steps)
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def time_stretch(self, data, speed_factor):
        stretch_flow = self.time_stretch_options[round(random.uniform(0, 1))]
        if stretch_flow == 'slower':
            speed_factor = 1 - speed_factor
        else:
            speed_factor = 1 + speed_factor
        new_sample = librosa.effects.time_stretch(data, speed_factor)
        # if size gets bigger or smaller than than the original
        if len(new_sample) > len(data):
            new_sample = new_sample[:len(data)]
        else:
            new_sample = np.append(new_sample, np.zeros((len(data)-len(new_sample),), dtype=int))
        return new_sample

    def db_shift(self, mel_spect, loudness):
        db_direction = self.pitch_shift_options[round(random.uniform(0, 1))]
        if db_direction == 'down':
            loudness = -loudness
        return mel_spect + loudness

    def augment(self, data, sampling_rate):
        # apply all augmentation operations
        # seconds
        shift_max = 5
        # 2 _> one tone
        pitch_factor = 2
        # faster (or slower)
        speed_factor = 0.5
        # decibels
        loudness = 10
        # Operations
        time_shifted = self.time_shift(data, sampling_rate, shift_max)
        pitch_shifted = self.pitch_shift(data, sampling_rate, pitch_factor)
        time_stretched = self.time_stretch(data, speed_factor)
        db_shifted = self.db_shift(librosa.power_to_db(librosa.feature.melspectrogram(data)), loudness)
        return data, time_shifted, pitch_shifted, time_stretched, db_shifted




