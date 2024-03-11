import tensorflow as tf
import keras

class MyCustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience, baseline):
        super(MyCustomEarlyStopping, self).__init__()
        self.initial_patience = patience
        self.patience = patience
        self.best = baseline
        self.best_weights = None

    def on_train_begin(self, logs=None):
        print('Start training!!!!!!!!!!!!!!')

    def on_epoch_end(self, epoch, logs=None):
        print(logs['val_loss'])
        if logs["val_loss"] < self.best:
            self.best = logs["val_loss"]
            self.patience = self.initial_patience
            self.best_weights = self.model.get_weights()
            print('Patience restored')
        else:
            self.patience -= 1
            print('Patience at {}'.format(self.patience))
            if self.patience == 0:
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.patience = self.initial_patience
                self.model.stop_training = True
