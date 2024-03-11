import time
import os
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

##################
## Options #########
##################
curr_path = Path('.').absolute().parent.__str__()
# Path to the directory with all necessary data
ptd = 'New_MERGE_Complete_Data/'
# Location of the .wav files converted from .mp3 as the intermediate step to generate Mel-spectrograms
location = 'total_wav/'

embeddings_file = 'embeddings_dataset_batched_original'  # 'embeddings_dataset_16kHz_padded'
samples_file = 'new_pub_complete_samples_normalized_16kHz'
#########################################

# Creates necessary directories to save the relevant metrics for evaluation
# as well as backup, should the process be terminated early.
def create_dirs():
    try:
        os.makedirs('time/')
    except FileExistsError:
        pass

    try:
        os.makedirs('f1_macro/')
    except FileExistsError:
        pass

    try:
        os.makedirs('f1_each/')
    except FileExistsError:
        pass

    try:
        os.makedirs('conf/')
    except FileExistsError:
        pass

    try:
        os.makedirs('accuracy/')
    except FileExistsError:
        pass

    return


# Trains and tests a Random Forest scikit-learn model with default parameters,
# using cross-validation to get 100 different train/test folds to more accurately
# assess the model's performance.
def run_rf_classifier(fwe, fwl):
    with open(fwe + '.pickle', 'rb') as emb_file:
        dataset = pickle.load(emb_file)
    with open(fwl + '.pickle', 'rb') as sounds_file:
        sounds = pickle.load(sounds_file)

    print(len(dataset))

    sounds.sort(key=lambda sounds: sounds[0])
    labels = np.array([temp[2] for temp in sounds])

    mapping = {}
    classes = ['Q1', 'Q2', 'Q3', 'Q4']
    for x in range(len(classes)):
        mapping[classes[x]] = x
    labels = np.array([mapping[temp] for temp in labels])
    dataset = np.asarray([data[0] for data in dataset])

    print(dataset.shape)
    print(labels.shape)
    print('Data ready!')

    create_dirs()
    time_dir, f1_macro_dir, f1_each_dir, conf_dir, accuracy_dir = \
        'time/', 'f1_macro/', 'f1_each/', 'conf/', 'accuracy/'

    print('Created dirs!')

    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    folds = list(kfold.split(dataset, labels))

    with open('folds_emb_rf.pickle', 'wb') as folds_file:
        pickle.dump(folds, folds_file)

    conf_global_matrix = np.zeros((4, 4))
    f1_macro_tot = []
    f1_each_tot = []

    acum = 0
    print('Start training...')
    for train, test in tqdm(folds):
        xtrain, ytrain = dataset[train], labels[train]
        xtest, ytest = dataset[test], labels[test]

        acum += 1

        start_time = time.time()

        rf_clf = RandomForestClassifier(n_jobs=46, random_state=1)
        rf_clf.fit(xtrain, ytrain)

        y_pred = rf_clf.predict(xtest)
        f1_macro = f1_score(ytest, y_pred, average='macro')
        f1_each = f1_score(ytest, y_pred, average=None)
        conf_temp = confusion_matrix(ytest, y_pred)
        acc = rf_clf.score(xtest, ytest)

        print('F1 Macro Score: {}'.format(str(f1_macro)))
        print('Confusion Matrix: {}'.format(str(conf_temp)))
        print('Acc: %0.4f' % acc)

        f1_macro_tot.append(f1_macro)
        f1_each_tot.append(f1_each)
        conf_global_matrix += conf_temp

        end_time = time.time()

        with open(time_dir + 'run_{}.txt'.format(acum), 'w') as time_f:
            time_f.write('Took {} minutes.'.format(round(end_time-start_time)/60))

        with open(f1_macro_dir + 'run_{}.pickle'.format(acum), 'wb') as f1_m_f:
            pickle.dump(f1_macro, f1_m_f)

        with open(f1_each_dir + 'run_{}.pickle'.format(acum), 'wb') as f1_e_f:
            pickle.dump(f1_each, f1_e_f)

        with open(conf_dir + 'run_{}.pickle'.format(acum), 'wb') as conf_f:
            pickle.dump(conf_temp, conf_f)

        with open(accuracy_dir + 'run_{}.pickle'.format(acum), 'wb') as acc_f:
            pickle.dump(acc, acc_f)

    f1_macro_tot = np.mean(f1_macro_tot)
    f1_each_tot = np.mean(f1_each_tot, axis=0)

    print('Final F1 Macro: {}'.format(f1_macro_tot))
    print('Final F1 Each: {}'.format(f1_each_tot))
    print('Global Conf Matrix: {}'.format(conf_global_matrix))

    with open('f1_macro_total.pickle', 'wb') as f1_mac_t:
        pickle.dump(np.mean(f1_macro_tot), f1_mac_t)

    with open('f1_each_total.pickle', 'wb') as f1_eac_t:
        pickle.dump(f1_each_tot, f1_eac_t)

    with open('conf_matrix_global.pickle', 'wb') as conf_t:
        pickle.dump(conf_global_matrix, conf_t)

    print('Done.')


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)

    run_rf_classifier(curr_path + ptd  + embeddings_file,
                                    curr_path + ptd + samples_file)
