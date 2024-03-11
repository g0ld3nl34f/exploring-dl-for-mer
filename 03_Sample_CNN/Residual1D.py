import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, \
                                    BatchNormalization, \
                                    MaxPooling1D, GlobalAveragePooling1D, ReLU
class Residual1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, pool_size=3, diff_size=False):
        super(Residual1D, self).__init__()

        self.conv = Conv1D(
            filters=filters, kernel_size=kernel_size, strides=stride, padding="same")
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp = MaxPooling1D(pool_size=pool_size)

    def call(self, input_tensor):
        output = self.mp(self.relu(self.bn(self.conv(input_tensor))))

        return output

    def get_config(self):
        cfg = super().get_config()
        return cfg