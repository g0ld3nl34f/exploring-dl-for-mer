import os
import pickle
import random
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np
import openl3
from natsort import natsorted

###################
## Options ##########
###################

curr_path = Path('.').absolute().parent.__str__()
# Path to the directory with all necessary data
ptd = 'New_MERGE_Complete_Data/'
# Location of the .wav files converted from .mp3 as the intermediate step to generate Mel-spectrograms
location = 'total_wav/'
# Name of the file with the song ids and respective quadrants
csv_name = 'new_pub_complete_targets.csv'
# Name of the file containing the samples embedded using the OpenL3 model.
embeddings_file = 'embeddings_dataset_batched_original'
# Name of file containing the time took to generate the embeddings.
embedding_time = 'embedding_time'

# Function that automates the process of generating embeddings for all
# samples using the music model from the OpenL3 framework.
# https://github.com/marl/openl3
def get_embeddings_one_one(path_to_files):
    unsorted_files = {f for f in listdir(location) if isfile(join(location, f))}
    to_sort = natsorted([path_to_files + f for f in unsorted_files if '.wav' in f])

    # Samples are embedded here. Decrease batch_size in process_audio_file
    # as necessary (low resources).
    print("Embedding {} wav files...".format(len(to_sort)))
    start_embedding = time.time()

    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)

    openl3.process_audio_file(to_sort, batch_size=32, verbose=True, model=model)

    end_embedding = time.time()
    time_elapsed = round((end_embedding - start_embedding) / 60)
    print("Done embedding.")

    # Samples are sorted and saved to the specified file name.
    unsorted_embeddings = {f for f in listdir(location) if isfile(join(location, f))}
    name_of_files = natsorted([f for f in unsorted_embeddings if '.npz' in f])
    embs = np.asarray([np.load(path_to_files + f)['embedding'] for f in name_of_files])

    with open(embeddings_file + ".pickle", 'wb') as emb_file:
        pickle.dump(embs, emb_file)

    with open(embedding_time + ".txt", 'w') as time_file:
        time_file.write("Embedding took {} minutes.".format(time_elapsed))

    return

# TODO: Check if the shape is correct
def fix_dataset():
    with open('embeddings_dataset_16kHz_padded' + '.pickle', 'rb') as emb_file:
        dataset = pickle.load(emb_file)

    fixed_dataset = []

    for samp in dataset:
        fixed_dataset.extend(samp[0])

    with open('embeddings_dataset_16kHz_padded_correct' + '.pickle', 'wb') as emb_file:
        pickle.dump(fixed_dataset, emb_file)

# This function may be used to check if the embeddings were sorted correctly.
def check_if_same(location):
    files = {f for f in listdir(location) if isfile(join(location, f))}

    old_files = [f for f in files]
    new_files = natsorted(old_files)

    for i in range(len(new_files)):
        print(str(old_files[i]) + ' ' + str(new_files[i]))


if __name__ == '__main__':
    random.seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    get_embeddings_one_one(curr_path + ptd + location)
    # fix_dataset()
    # check_if_same(location)
