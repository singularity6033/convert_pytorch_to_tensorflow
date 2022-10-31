import numpy as np

import os
import time
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization


model_save_path = './sgd_tf2_ndn_best/model'
weights_save_path = './sgd_tf2_ndn_best/weights'

block10_0 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_0_sgd_ndn.pt').cpu().data.numpy().T
block10_2 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_1_sgd_ndn.pt').cpu().data.numpy().T
block10_4 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_2_sgd_ndn.pt').cpu().data.numpy().T
block10_5 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_3_sgd_ndn.pt').cpu().data.numpy().T
block10_7 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_4_sgd_ndn.pt').cpu().data.numpy().T
block10_8 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_5_sgd_ndn.pt').cpu().data.numpy().T
block10_10 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_6_sgd_ndn.pt').cpu().data.numpy().T
block10_11 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_7_sgd_ndn.pt').cpu().data.numpy().T
block10_14 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_8_sgd_ndn.pt').cpu().data.numpy().T
block10_15 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_9_sgd_ndn.pt').cpu().data.numpy().T
block10_16 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_10_sgd_ndn.pt').cpu().data.numpy().T
block10_17 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_11_sgd_ndn.pt').cpu().data.numpy().T
block10_18 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_12_sgd_ndn.pt').cpu().data.numpy().T
block10_19 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_13_sgd_ndn.pt').cpu().data.numpy().T
block10_20 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_14_sgd_ndn.pt').cpu().data.numpy().T
block10_21 = torch.load('./vgg16_cifar100_sgd_ndn_best/vgg16_cifar100_15_sgd_ndn.pt').cpu().data.numpy().T


# model architecture
model = Sequential()
# The First Block
model.add(Conv2D(64, (3, 3), padding='same', input_shape=[32, 32, 3]))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', input_shape=[32, 32, 3]))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The Second Block
model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The Third Block
model.add(Conv2D(256, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
#
model.add(Conv2D(256, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The 4th Block
model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
#
model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The 5th Block
model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
#
model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
model.add(Flatten())
model.add(Dense(512, name='dense_1'))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(512, name='dense_2'))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(100, name='dense_3'))
model.add(Activation('softmax'))

model.summary()
model_json = model.to_json()
with open(os.path.sep.join([model_save_path, "sgd_vgg16_cifar100_-11.json"]), "w") as json_file:
    json_file.write(model_json)
#
model.layers[0].set_weights([block10_0, np.zeros(64)])
model.layers[2].set_weights([block10_2, np.zeros(64)])
model.layers[5].set_weights([block10_4, np.zeros(128)])
model.layers[7].set_weights([block10_5, np.zeros(128)])
model.layers[10].set_weights([block10_7, np.zeros(256)])
model.layers[12].set_weights([block10_8, np.zeros(256)])
model.layers[14].set_weights([block10_10, np.zeros(256)])
model.layers[17].set_weights([block10_11, np.zeros(512)])
model.layers[19].set_weights([block10_14, np.zeros(512)])
model.layers[21].set_weights([block10_15, np.zeros(512)])
model.layers[24].set_weights([block10_16, np.zeros(512)])
model.layers[26].set_weights([block10_17, np.zeros(512)])
model.layers[28].set_weights([block10_18, np.zeros(512)])
model.layers[32].set_weights([block10_19, np.zeros(512)])
model.layers[34].set_weights([block10_20, np.zeros(512)])
model.layers[36].set_weights([block10_21, np.zeros(100)])
#
model.save(os.path.join(weights_save_path, 'sgd_vgg16_cifar100.h5'))
