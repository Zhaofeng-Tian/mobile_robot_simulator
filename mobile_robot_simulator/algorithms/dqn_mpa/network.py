import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow as tf

class DeepQNetwork(keras.Model):
    def __init__(self, input_dims, state_shape,n_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=(4, 4), activation='relu',
                            input_shape=input_dims)
        self.conv2 = Conv2D(64, 4, strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, 3, strides=(1, 1), activation='relu')
        
        self.conv4 = Conv2D(32, 8, strides=(4, 4), activation='relu',
                            input_shape=input_dims)
        self.conv5 = Conv2D(64, 4, strides=(2, 2), activation='relu')
        self.conv6 = Conv2D(64, 3, strides=(1, 1), activation='relu')
        self.flat1 = Flatten()
        self.flat2 = Flatten()
        self.fc1 = Dense(512, activation='relu')
        # self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(n_actions, activation=None)

    def call(self, inputs):
        #print("Q-net in call")
        x,l,s = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        l = self.conv4(l)
        l = self.conv5(l)
        l = self.conv6(l)

        x = self.flat1(x)
        l = self.flat2(l)
        x = tf.concat([x,l,s],1)
        x = self.fc1(x)
        x = self.fc2(x)
        #print("Q-net output: ", str(x))
        return x
