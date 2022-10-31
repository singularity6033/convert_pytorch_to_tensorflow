import numpy as np

import os
import time
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization


model_save_path = './vgg16_cifar10_fpl_tf2/model'
weights_save_path = './vgg16_cifar10_fpl_tf2/weights'

# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w1_0.pt').cpu().data.numpy().T
# block_2 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w1_2.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w2_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w2_1.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w2_4.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w3_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w3_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w3_3.pt').cpu().data.numpy().T
# block_5 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w3_5.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w4_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w4_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w4_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w4_4.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w4_7.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w5_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w5_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w5_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w5_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w5_6.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w5_8.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w6_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w6_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w6_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w6_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w6_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w6_7.pt').cpu().data.numpy().T
# block_9 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w6_9.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_8.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w7_11.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_10.pt').cpu().data.numpy().T
# block_12 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w8_12.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_10.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_11.pt').cpu().data.numpy().T
# block_13 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w9_13.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_10.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_11.pt').cpu().data.numpy().T
# block_12 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_12.pt').cpu().data.numpy().T
# block_15 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w10_15.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_10.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_11.pt').cpu().data.numpy().T
# block_12 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_12.pt').cpu().data.numpy().T
# block_14 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_14.pt').cpu().data.numpy().T
# block_16 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w11_16.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_10.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_11.pt').cpu().data.numpy().T
# block_12 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_12.pt').cpu().data.numpy().T
# block_14 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_14.pt').cpu().data.numpy().T
# block_15 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_15.pt').cpu().data.numpy().T
# block_17 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w12_17.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_10.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_11.pt').cpu().data.numpy().T
# block_12 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_12.pt').cpu().data.numpy().T
# block_14 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_14.pt').cpu().data.numpy().T
# block_15 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_15.pt').cpu().data.numpy().T
# block_16 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_16.pt').cpu().data.numpy().T
# block_19 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w13_19.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_10.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_11.pt').cpu().data.numpy().T
# block_12 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_12.pt').cpu().data.numpy().T
# block_14 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_14.pt').cpu().data.numpy().T
# block_15 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_15.pt').cpu().data.numpy().T
# block_16 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_16.pt').cpu().data.numpy().T
# block_19 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_19.pt').cpu().data.numpy().T
# block_20 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w14_20.pt').cpu().data.numpy().T
#
# block_0 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_0.pt').cpu().data.numpy().T
# block_1 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_1.pt').cpu().data.numpy().T
# block_3 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_3.pt').cpu().data.numpy().T
# block_4 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_4.pt').cpu().data.numpy().T
# block_6 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_6.pt').cpu().data.numpy().T
# block_7 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_7.pt').cpu().data.numpy().T
# block_8 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_8.pt').cpu().data.numpy().T
# block_10 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_10.pt').cpu().data.numpy().T
# block_11 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_11.pt').cpu().data.numpy().T
# block_12 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_12.pt').cpu().data.numpy().T
# block_14 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_14.pt').cpu().data.numpy().T
# block_15 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_15.pt').cpu().data.numpy().T
# block_16 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_16.pt').cpu().data.numpy().T
# block_19 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_19.pt').cpu().data.numpy().T
# block_20 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_20.pt').cpu().data.numpy().T
# block_21 = torch.load('./vgg16_cifar10_fpl/cifar10_vgg16_w15_21.pt').cpu().data.numpy().T

# model architecture
model = Sequential()
# The First Block
model.add(Conv2D(64, (3, 3), padding='same', input_shape=[32, 32, 3]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', input_shape=[32, 32, 3]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The Second Block
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The Third Block
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The 4th Block
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The 5th Block
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
# model_json = model.to_json()
# with open(os.path.sep.join([model_save_path, "s15_-9.json"]), "w") as json_file:
#     json_file.write(model_json)
#
# model.layers[0].set_weights([block_0, np.zeros(64)])
# model.layers[2].set_weights([block_1, np.zeros(64)])
# model.layers[5].set_weights([block_3, np.zeros(128)])
# model.layers[7].set_weights([block_4, np.zeros(128)])
# model.layers[10].set_weights([block_6, np.zeros(256)])
# model.layers[12].set_weights([block_7, np.zeros(256)])
# model.layers[14].set_weights([block_8, np.zeros(256)])
# model.layers[17].set_weights([block_10, np.zeros(512)])
# model.layers[19].set_weights([block_11, np.zeros(512)])
# model.layers[21].set_weights([block_12, np.zeros(512)])
# model.layers[24].set_weights([block_14, np.zeros(512)])
# model.layers[26].set_weights([block_15, np.zeros(512)])
# model.layers[28].set_weights([block_16, np.zeros(512)])
# model.layers[32].set_weights([block_19, np.zeros(512)])
# model.layers[34].set_weights([block_20, np.zeros(512)])
# model.layers[-2].set_weights([block_21, np.zeros(10)])
# #
# model.save(os.path.join(weights_save_path, 'w15.h5'))
