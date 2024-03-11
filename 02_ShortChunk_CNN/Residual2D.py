import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import relu


class Residual2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=(2,2), diff_size=False):
        super(Residual2D, self).__init__()

        self.conv_1 = Conv2D(
            filters=filters, kernel_size = (kernel_size, kernel_size), strides=stride, padding='same')
        self.bn_1 = BatchNormalization()

        self.conv_2 = Conv2D(
            filters=filters, kernel_size=(kernel_size, kernel_size), padding='same')
        self.bn_2 = BatchNormalization()

        self.diff = False
        if (stride != 1) or diff_size:
            self.conv_3 = Conv2D(
                filters=filters, kernel_size=(kernel_size, kernel_size), strides=stride, padding='same')
            self.bn_3 = BatchNormalization()
            self.diff = True

    def call(self, input_tensor):
        output = self.conv_1(input_tensor)
        output = self.bn_1(output)
        output = relu(output)

        output = self.conv_2(output)
        output = self.bn_2(output)

        if self.diff:
            input_tensor = self.conv_3(input_tensor)
            input_tensor = self.bn_3(input_tensor)

        output += input_tensor
        output = relu(output)
        return output

    def get_config(self):
        cfg = super().get_config()
        return cfg
