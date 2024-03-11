import tensorflow as tf


class MyCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyCallbacks, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        acc = logs["accuracy"]
        if acc >= self.threshold:
            self.model.stop_training = True
