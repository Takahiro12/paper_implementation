import tensorflow as tf
import config

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        if strides > 1: # stridesを変えると、公式より入力次元と出力次元が異なる。以下のダウンサンプリングで出力次元を合わせる
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x


    def call(self, x, training=None):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.keras.layers.add([residual, x])
        x = tf.nn.relu(x)
        return x
        

class Resnet18(tf.keras.Model):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.layer2_1 = BasicBlock(filters=64)
        self.layer2_2 = BasicBlock(filters=64)
        self.layer3_1 = BasicBlock(filters=128, strides=2) 
        self.layer3_2 = BasicBlock(filters=128)
        self.layer4_1 = BasicBlock(filters=256, strides=2)
        self.layer4_2 = BasicBlock(filters=256)
        self.layer5_1 = BasicBlock(filters=512, strides=2)
        self.layer5_2 = BasicBlock(filters=512)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(config.NUM_CLASSES, activation=tf.keras.activations.softmax)
    
    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer2_1(x, training=training)
        x = self.layer2_2(x, training=training)
        x = self.layer3_1(x, training=training)
        x = self.layer3_2(x, training=training)
        x = self.layer4_1(x, training=training)
        x = self.layer4_2(x, training=training)
        x = self.layer5_1(x, training=training)
        x = self.layer5_2(x, training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x



