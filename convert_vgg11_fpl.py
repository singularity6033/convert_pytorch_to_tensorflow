import numpy as np

import os
import time
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization


# model_save_path = './vgg11_cifar100_fpl_tf2/model'
# weights_save_path = './vgg11_cifar100_fpl_tf2/weights'
#
# block1_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w1_0.pt').cpu().data.numpy().T
# block1_3 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w1_3.pt').cpu().data.numpy().T
#
# block2_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w2_0.pt').cpu().data.numpy().T
# block2_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w2_2.pt').cpu().data.numpy().T
# block2_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w2_5.pt').cpu().data.numpy().T
#
# block3_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w3_0.pt').cpu().data.numpy().T
# block3_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w3_2.pt').cpu().data.numpy().T
# block3_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w3_4.pt').cpu().data.numpy().T
# block3_6 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w3_6.pt').cpu().data.numpy().T
#
# block4_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w4_0.pt').cpu().data.numpy().T
# block4_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w4_2.pt').cpu().data.numpy().T
# block4_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w4_4.pt').cpu().data.numpy().T
# block4_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w4_5.pt').cpu().data.numpy().T
# block4_8 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w4_8.pt').cpu().data.numpy().T
#
# block5_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w5_0.pt').cpu().data.numpy().T
# block5_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w5_2.pt').cpu().data.numpy().T
# block5_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w5_4.pt').cpu().data.numpy().T
# block5_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w5_5.pt').cpu().data.numpy().T
# block5_7 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w5_7.pt').cpu().data.numpy().T
# block5_9 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w5_9.pt').cpu().data.numpy().T
#
# block6_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w6_0.pt').cpu().data.numpy().T
# block6_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w6_2.pt').cpu().data.numpy().T
# block6_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w6_4.pt').cpu().data.numpy().T
# block6_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w6_5.pt').cpu().data.numpy().T
# block6_7 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w6_7.pt').cpu().data.numpy().T
# block6_8 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w6_8.pt').cpu().data.numpy().T
# block6_11 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w6_11.pt').cpu().data.numpy().T
#
# block7_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_0.pt').cpu().data.numpy().T
# block7_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_2.pt').cpu().data.numpy().T
# block7_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_4.pt').cpu().data.numpy().T
# block7_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_5.pt').cpu().data.numpy().T
# block7_7 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_7.pt').cpu().data.numpy().T
# block7_8 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_8.pt').cpu().data.numpy().T
# block7_10 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_10.pt').cpu().data.numpy().T
# block7_12 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w7_12.pt').cpu().data.numpy().T
#
# block8_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_0.pt').cpu().data.numpy().T
# block8_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_2.pt').cpu().data.numpy().T
# block8_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_4.pt').cpu().data.numpy().T
# block8_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_5.pt').cpu().data.numpy().T
# block8_7 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_7.pt').cpu().data.numpy().T
# block8_8 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_8.pt').cpu().data.numpy().T
# block8_10 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_10.pt').cpu().data.numpy().T
# block8_11 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_11.pt').cpu().data.numpy().T
# block8_14 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w8_14.pt').cpu().data.numpy().T
#
# block9_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_0.pt').cpu().data.numpy().T
# block9_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_2.pt').cpu().data.numpy().T
# block9_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_4.pt').cpu().data.numpy().T
# block9_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_5.pt').cpu().data.numpy().T
# block9_7 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_7.pt').cpu().data.numpy().T
# block9_8 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_8.pt').cpu().data.numpy().T
# block9_10 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_10.pt').cpu().data.numpy().T
# block9_11 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_11.pt').cpu().data.numpy().T
# block9_14 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_14.pt').cpu().data.numpy().T
# block9_15 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w9_15.pt').cpu().data.numpy().T
#
# block10_0 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_0.pt').cpu().data.numpy().T
# block10_2 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_2.pt').cpu().data.numpy().T
# block10_4 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_4.pt').cpu().data.numpy().T
# block10_5 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_5.pt').cpu().data.numpy().T
# block10_7 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_7.pt').cpu().data.numpy().T
# block10_8 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_8.pt').cpu().data.numpy().T
# block10_10 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_10.pt').cpu().data.numpy().T
# block10_11 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_11.pt').cpu().data.numpy().T
# block10_14 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_14.pt').cpu().data.numpy().T
# block10_15 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_15.pt').cpu().data.numpy().T
# block10_16 = torch.load('./vgg11_cifar100_fpl/cifar100_vgg11_w10_16.pt').cpu().data.numpy().T


# model architecture
model = Sequential()
# The First Block
model.add(Conv2D(64, (3, 3), padding='same', input_shape=[32, 32, 3]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The Second Block
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The Third Block
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
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# # The 5th Block
# model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(Activation('relu'))
#
# model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dense(512))
# model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('softmax'))

model.summary()
# model_json = model.to_json()
# with open(os.path.sep.join([model_save_path, "s1_-5.json"]), "w") as json_file:
#     json_file.write(model_json)
#
# model.layers[0].set_weights([block1_0, np.zeros(64)])
# model.layers[-2].set_weights([block1_3, np.zeros(100)])
# model.layers[-2].set_weights([block2_5, np.zeros(10)])
# model.layers[-2].set_weights([block3_6, np.zeros(100)])
# model.layers[-2].set_weights([block4_8, np.zeros(10)])
# model.layers[-2].set_weights([block5_9, np.zeros(100)])
# model.layers[-2].set_weights([block6_11, np.zeros(10)])
# model.layers[-2].set_weights([block7_12, np.zeros(100)])
# model.layers[-2].set_weights([block8_14, np.zeros(10)])
# model.layers[-2].set_weights([block9_15, np.zeros(100)])
# model.layers[-2].set_weights([block10_16, np.zeros(10)])
#
# model.save(os.path.join(weights_save_path, 'w1.h5'))
