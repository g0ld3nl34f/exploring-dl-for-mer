import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU

class Conv2dRegular(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), stride=1, pooling=(2,2)):
        super(Conv2dRegular, self).__init__()
        self.conv = Conv2D(filters, kernel_size, stride, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp = MaxPool2D(pooling)
    def call(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

