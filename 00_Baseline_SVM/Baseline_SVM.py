import sys
sys.path.append('../Utils/')
import random
import time
import pickle
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.svm import *
from sklearn.model_selection import *
from sklearn.metrics import *
from keras.utils import np_utils

# manually tune svm model hyperparameters using skopt on the ionosphere dataset

from pathlib import Path
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

from tqdm import tqdm

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
samples_file = 'top_100_new_pub_complete.csv'

###############
## Utils ######
###############

def load_data():
    # carregar .csv com top 100
    data = pd.read_csv(
        curr_path + ptd + samples_file,
        sep=',', header=0)
    data.sort_values(by=['IDSong'])
    X = data.iloc[:, 1:101]  # cabeçalho, 1a coluna -> nome, label no fim

    target = pd.read_csv(
        curr_path + ptd + csv_name,
        sep=',', header=0)
    target.sort_values(by=['SongID'])
    Y = target.iloc[:, 1]

    # normalizar dataset
    X = preprocessing.scale(X.to_numpy())

    return X, Y


def avg(lst):
    return sum(lst) / len(lst)


def map_targets(Y):
    # converter para inteiros
    mapping = {}
    classes = ['Q1', 'Q2', 'Q3', 'Q4']
    for x in range(len(classes)):
        mapping[classes[x]] = x
    Y_hot_encoded = [mapping[temp] for temp in Y]
    # one hot enconding
    Y_hot_encoded = np_utils.to_categorical(Y_hot_encoded)
    return Y_hot_encoded


##########################
## Find Hyperparameters ##
##########################

def find_optimal_hyperparameters(X, y):
    # define the space of hyperparameters to search
    search_space = list()
    search_space.append(Real(1e-6, 100.0, 'log-uniform', name='C'))
    search_space.append(Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'))
    search_space.append(Integer(1, 5, name='degree'))
    search_space.append(Real(1e-6, 100.0, 'log-uniform', name='gamma'))


    # define the function used to evaluate a given configuration
    @use_named_args(search_space)
    def evaluate_model(**params):
        # configure the model with specific hyperparameters
        model = SVC()
        model.set_params(**params)
        # define test harness
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
        # n_jobs has to be 1, or this never completes
        result = cross_val_score(model, X, y, cv=cv, n_jobs=1, scoring='accuracy', verbose=1)
        # calculate the mean of the scores
        estimate = np.mean(result)
        # convert from a maximizing score to a minimizing score
        return 1.0 - estimate

    # load dataset
    print(X.shape, y.shape)
    # perform optimization
    result = gp_minimize(evaluate_model, search_space, verbose=1, random_state=1, n_jobs=20)
    # summarizing finding:
    print('Best Accuracy: %.3f' % (1.0 - result.fun))
    print('Best Parameters: %s' % result.x)

    with open('new_pub_complete_find_hp_results.txt', 'w') as res:
        res.write('Best Accuracy: %.3f\n' % (1.0 - result.fun))
        res.write('Best Parameters: %s' % result.x)

    return result.x


#######################################
## Run SVM with found Hyperparamters ##
#######################################

def run_SVM(X, y, Y_string, c, kernel, degree, gamma):
    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    macro_f1_score, f1_each_all, conf_mat_global = [], [], np.zeros((4, 4))

    optimal_list = find_optimal_hyperparameters(X, y)

    start_time = time.time()
    for train, test in tqdm(list(kfold.split(X, Y_string))):
        svc = SVC(C=optimal_list['C'],
                  kernel=optimal_list['Kernel'],
                  degree=optimal_list['Degree'],
                  gamma=optimal_list['Gamma'])
        svc.fit(X[train], y[train])

        ypred = svc.predict(X[test])
        f1_temp = f1_score(y[test], ypred, average='macro')
        f1_each = f1_score(y[test], ypred, average=None)
        conf_mat = confusion_matrix(y[test], ypred)

        macro_f1_score.append(f1_temp)
        f1_each_all.append(f1_each)
        conf_mat_global += conf_mat

    print(macro_f1_score)
    print("\nAVG F1_Score: " + str(avg(macro_f1_score)))
    end_time = time.time() - start_time

    # save macro f1
    with open("f1_score_svm_100_features.pickle", "wb") as file_results:
        pickle.dump(macro_f1_score, file_results)
    with open("f1_each_svm_100_features.pickle", "wb") as file_each_results:
        pickle.dump(f1_each_all, file_each_results)
    with open("conf_mat_svm_100_features.pickle", "wb") as conf_mat_results:
        pickle.dump(conf_mat_global, conf_mat_results)
    with open("time_svm_100_features.txt", "w") as time_file:
        time_file.write('Took {} minutes to run.'.format(end_time))

    return


def main():
    # load data
    X, Y = load_data()
    print("Dados carregados com sucesso!")
    # hot one encoding
    Y_encoded = map_targets(Y)

    start = time.time()
    # Not used for baseline reasons
    result = find_optimal_hyperparameters(X, Y)
    with open('best_hyper_for_corrcted.pickle', 'wb') as hype_file:
        pickle.dump(result, hype_file)
    end = time.time()
    time_elapsed = round(end - start)
    print("Minutos passados: " + str(time_elapsed / 60))

    run_SVM(X, Y, Y,
                         result.get('C'), result.get('rbf'),
                         result.get('degree'), result.get('gamma'))

    print('Finished')


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    main()
