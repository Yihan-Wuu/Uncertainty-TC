from __future__ import print_function
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, \
    Conv2D, MaxPooling2D, BatchNormalization, GaussianNoise, Multiply, GlobalAveragePooling2D, Add
from tensorflow.keras.regularizers import l1_l2, L1
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg16 import VGG16

from tqdm import tqdm
import tensorflow as tf

# tf.config.experimental.f
from PIL import Image
import os
#import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import argparse

import sys
sys.path.append("../tools/")



def plot_history(history, fig_name, ignore_num=0, show = False):
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    # acc_values = history_dict['mse']
    # val_acc_values = history_dict['val_mse']

    epochs = range(1, len(loss_values) + 1 -ignore_num)

    plt.plot(epochs, loss_values[ignore_num:], 'bo', label='Training loss')#bo:blue dot蓝点
    plt.plot(epochs, val_loss_values[ignore_num:], 'ro', label='Validation loss')#b: blue蓝色
    #plt.plot(epochs, acc_values[ignore_num:], 'b', label='Training mae')#bo:blue dot蓝点
    #plt.plot(epochs, val_acc_values[ignore_num:], 'r-', label='Validation mae')#b: blue蓝色
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig = plt.gcf() # plt.savefig(fig_name)
    if show == True:
        plt.show()
    fig.savefig(fig_name, dpi=100)
    plt.close()

def rotate_by_channel(data, sita, length=2):
    newdata = []
    chanel_num = data.shape[3]
    height = data.shape[1]
    if length > 1:
        for index, singal in enumerate(data):
            new_sam = np.array([])
            for i in range(chanel_num):
                channel = singal[:,:,i]
                img = Image.fromarray(channel)
                new_img = img.rotate(sita[index])
                new_channel = np.asarray(new_img)
                if i==0:
                    new_sam = new_channel
                else:
                    new_sam = np.concatenate((new_sam, new_channel), axis = 1) 
            new_sam = new_sam.reshape((height,height,chanel_num),order='F')
            newdata.append(new_sam)
    else:
        print("Error! data length = 1...")
    return np.array(newdata)

def AlexNet(W_l1RE, W_l2RE, shape, dropout_net=0):
    model = Sequential() # 16 32 64 128    256 64 1
    model.add(Conv2D(16, (4, 4), strides = 2, padding='valid',
                     input_shape=shape,
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    #model.add(AveragePooling2D((2, 2), strides = 1))
    model.add(Conv2D(32, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3) , strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_net))
    model.add(Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_net))
    model.add(Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))


    opt = keras.optimizers.RMSprop(lr=0.005)

    # Let's train the model using RMSprop
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    model.summary()
    return model

def vgg1(W_l1RE, W_l2RE, shape, dropout_net=0):
    img_input = Input(shape=shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=L1(W_l1RE))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=L1(W_l1RE))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=L1(W_l1RE))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=L1(W_l1RE))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=L1(W_l1RE))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=L1(W_l1RE))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout_net)(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(dropout_net)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(dropout_net)(x)

    x = Dense(1, activation='linear', name='predictions')(x)

    model = Model(inputs=[img_input], outputs=[x])
    # decay是学习率衰减
    opt = keras.optimizers.RMSprop(lr=0.005)

    model.compile(loss=keras.losses.mse, optimizer=opt, metrics=['mse'])
    model.summary()
    return model

def vgg(W_l1RE, W_l2RE, shape, dropout_net):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     input_shape=shape, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)
                     , kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(dropout_net))
    model.add(Dense(64, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dropout(dropout_net))
    model.add(Dense(1, kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('linear'))

    opt = keras.optimizers.RMSprop(lr=0.005)

    # Let's train the model using RMSprop
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    model.summary()
    return model

def Network_classification_Dense_dropout(W_l1RE, shape, dropout_net=0.5):
    inputs = Input(shape=shape)
    x = Conv2D(64, (10, 10), strides=1, padding='valid',
               kernel_regularizer=L1(W_l1RE))(inputs)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(256, (5, 5), strides=1, dilation_rate=(2, 2), kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(288, (3, 3), strides=1, padding='same', dilation_rate=(2, 2), kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=1)(x)

    x = Conv2D(272, (3, 3), strides=1, padding='same', dilation_rate=(2, 2), kernel_regularizer=L1(W_l1RE))(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=1, padding='same', kernel_regularizer=L1(W_l1RE))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(3586, kernel_regularizer=L1(W_l1RE))(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_net)(x)
    x = Dense(2048, kernel_regularizer=L1(W_l1RE))(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_net)(x)
    x = Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
              kernel_regularizer=L1(W_l1RE))(x)
    output = Activation('linear')(x)
    model = Model(inputs=[inputs], outputs=[output])
    # decay是学习率衰减
    opt = keras.optimizers.Adam(lr=0.005)

    model.compile(loss=keras.losses.mse, optimizer=opt, metrics=['mse'])
    model.summary()
    return model

def branch_net(input_shape, name_idx, dropout=0.5):
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    conv_block = Sequential(name=f'conv_block_{name_idx}')
    conv_block.add(Conv2D(8, (3, 3), 1, padding='same', activation='relu'))
    conv_block.add(MaxPooling2D((2, 2), 2))
    conv_block.add(Conv2D(32, (3, 3), 1, padding='same', activation='relu'))
    conv_block.add(MaxPooling2D((2, 2), 2))

    wv_block = Sequential(name=f'WV_block_{name_idx}')
    wv_block.add(Conv2D(32, (10, 10), strides=2, padding='same', activation='relu'))
    wv_block.add(MaxPooling2D((2, 2), 2))

    extractor = Sequential(name=f'Extractor_{name_idx}')
    extractor.add(Conv2D(128, (3, 3), 1, padding='same', activation='relu'))
    extractor.add(Conv2D(128, (3, 3), 1, padding='same', activation='relu'))
    extractor.add(MaxPooling2D((2, 2), 2))
    extractor.add(Conv2D(256, (3, 3), 1, padding='same', activation='relu'))
    extractor.add(Conv2D(256, (3, 3), 1, padding='same', activation='relu'))
    extractor.add(GlobalAveragePooling2D())

    regressor = Sequential(name=f'regressor_{name_idx}')
    regressor.add(Dense(2048, activation='relu'))
    regressor.add(Dropout(dropout))
    regressor.add(Dense(2048, activation='relu'))
    regressor.add(Dropout(dropout))
    regressor.add(Dense(166, activation='softmax'))

    conv1 = conv_block(input1)
    atten1 = wv_block(input2)

    feat1 = Multiply()([conv1, atten1])

    feat1 = extractor(feat1)

    feat1 = Flatten()(feat1)

    prob1 = regressor(feat1)

    vector = tf.convert_to_tensor(np.concatenate([
        np.arange(13, 17.3, 0.3),
        np.arange(17.3, 24.6, 0.6),
        np.arange(24.6, 51.1, 0.3),
        np.arange(51.1, 79.9, 0.6)
    ]).reshape(-1, 1), dtype='float')

    out1 = keras.backend.dot(prob1, vector)

    return input1, input2, out1

def custom_loss(y_true, y_pred):
    y_c, y_p = y_pred[0], y_pred[1]
    loss_1 = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_c))) \
             + tf.reduce_mean(tf.abs(y_true - y_c) / y_c)
    loss_2 = 0.5 * tf.reduce_mean(tf.abs(y_c - y_p) - 0.5)

    return loss_1 + loss_2

def siamese_net(input_shape, dropout):
    input1, input2, out1 = branch_net(input_shape, 0, dropout)
    input3, input4, out2 = branch_net(input_shape, 1, dropout)
    out = Add()([out1, out2]) / 2
    model = Model(inputs=[input1, input2, input3, input4], outputs=[out])
    model.summary()
    model.compile(loss=['mse'], optimizer=keras.optimizers.RMSprop(learning_rate=0.0001), metrics=['mae'])

    return model
# model = siamese_net((60, 60, 1))

def normalize_data(x_test, chanel_num):
    result=[]
    height = x_test.shape[1]
    for each_sam in x_test:
        new_sam = []
        for i in range(chanel_num):
            chanel = each_sam[:,:,i]
            chanel = (chanel - np.mean(chanel)) / (np.std(chanel)+0.01)
            if i==0:
                new_sam = chanel
            else:
                new_sam = np.concatenate((new_sam, chanel), axis =1)
               
        new_sam = new_sam.reshape((height,height,chanel_num),order='F')
        result.append(new_sam)
    result = np.array(result)
    return result

def train_siamese_net(EPOCHS, trainset_xpath, trainset_ypath, testset_xpath, testset_ypath, dropout_data=0, dropout_net=0, idx=0):
    batch_size = 64
    epochs = EPOCHS
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'result_model')
    
    x_train = np.load(trainset_xpath).astype('float32')

    y_train = np.load(trainset_ypath).astype('float32')
    x_test = np.load(testset_xpath).astype('float32')

    y_test = np.load(testset_ypath).astype('float32')

    x_test = x_test[y_test <= 180, :, :, :]
    y_test = y_test[y_test <= 180]
    x_train = x_train[:, 20:80, 20:80, :]   # 18:83 = 65
    x_train = normalize_data(x_train, x_train.shape[3])
    x_train1 = np.expand_dims(x_train[:, :, :, 0], -1)
    x_train2 = np.expand_dims(x_train[:, :, :, 1], -1)
    x_test = x_test[:, 20:80, 20:80, :]   # 18:82 = 64
    x_test = normalize_data(x_test, x_test.shape[3])
    x_test1 = np.expand_dims(x_test[:, :, :, 0], -1)
    x_test2 = np.expand_dims(x_test[:, :, :, 1], -1)

    print("the shape of train set and test set: ", x_train.shape, x_test.shape)
    model_name_pre = 'Siamese-'
    model = siamese_net((60, 60, 1), dropout_net)
    print(x_train1.shape)

    if not data_augmentation:
        history = model.fit((x_train1[:-1], x_train2[:-1], x_train1[1:], x_train2[1:]), y_train[1:],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=((x_test1[:-1], x_test2[:-1], x_test1[1:], x_test2[1:]), y_test[1:]),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
    
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=20, min_lr=1e-6)
        #tb = TensorBoard(log_dir='./tmp/log', histogram_freq=10)
        filepath="./NASA_model/siamese_dropout_data={}_dropout_net={}".format(
            dropout_data, dropout_net)+".hdf5"
        checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1,  period=10, save_weights_only=True)

        datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True)

        def generate_data_generator(X1, X2, X3, X4, Y):
            genX1 = datagen.flow(X1, Y, batch_size=batch_size, seed=123)
            genX2 = datagen.flow(X2, Y, batch_size=batch_size, seed=123)
            genX3 = datagen.flow(X3, Y, batch_size=batch_size, seed=123)
            genX4 = datagen.flow(X4, Y, batch_size=batch_size, seed=123)
            while True:
                x1, y = genX1.next()
                x2, y = genX2.next()
                x3, y = genX3.next()
                x4, y = genX4.next()
                yield [x1, x2, x3, x4], y

        history = model.fit_generator(generate_data_generator(x_train1[:-1], x_train2[:-1], x_train1[1:],
                                                                              x_train2[1:], y_train[1:]),#Mygen(x_train, y_train, batch_size=batch_size),
                                      epochs=epochs,
                                      validation_data=((x_test1[:-1], [x_test2[:-1], x_test1[1:], x_test2[1:]]), y_test[1:]),
                                      shuffle=True,
                                      verbose=2,
                                      steps_per_epoch=int(x_train.shape[0]/batch_size)+1,
                                      workers=8,
                                      use_multiprocessing=True,
                                      callbacks=[reduce_lr, checkpoint])

    scores = model.evaluate((x_test1[:-1], x_test2[:-1], x_test1[1:], x_test2[1:]), y_test[1:], verbose=1)
    print('Test RMSE:', np.sqrt(scores[0]))
    print('Test MAE:', scores[1])
    model_name = model_name_pre + '_dropout_data={}_dropout_net={}_{}'.format(dropout_data,
        dropout_net, idx) + 'rotate-RMSE' + str(int(np.sqrt(scores[0])*100)/100.0) + '.h5'
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    #plot_history(history, model_path+"-"+str(int(np.sqrt(scores[0])*100)/100.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    plot_history(history, model_path+"-"+str(int(scores[1]*10)/10.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    print("Over!!!")
    return model_path


def train_net(model_name: str, EPOCHS, trainset_xpath, trainset_ypath, testset_xpath, testset_ypath, dropout_data=0, dropout_net=0,
                  idx=0):
    batch_size = 64
    epochs = EPOCHS
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'result_model')

    x_train = np.load(trainset_xpath).astype('float32')

    y_train = np.load(trainset_ypath).astype('float32')
    x_test = np.load(testset_xpath).astype('float32')

    y_test = np.load(testset_ypath).astype('float32')

    x_test = x_test[y_test <= 180, :, :, :]
    y_test = y_test[y_test <= 180]
    x_train = x_train[:, 18:83, 18:83, :]  # 18:83 = 65
    x_train = normalize_data(x_train, x_train.shape[3])
    x_test = x_test[:, 18:83, 18:83, :]  # 18:82 = 64
    x_test = normalize_data(x_test, x_test.shape[3])

    print("the shape of train set and test set: ", x_train.shape, x_test.shape)
    model_name_pre = 'Sel_PostNet-'
    if model_name.lower() == 'alexnet':
        model = AlexNet(W_l1RE=1e-4, W_l2RE=0, shape=(65, 65, 2), dropout_net=dropout_net)
    elif model_name.lower() == 'vgg':
        model = vgg(W_l1RE=1e-4, W_l2RE=0, shape=(65, 65, 2), dropout_net=dropout_net)
    else:
        model = Network_classification_Dense_dropout(W_l1RE=1e-4, shape=(65, 65, 2), dropout_net=dropout_net)
    print(x_train.shape)

    if not data_augmentation:
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_data=(x_test, y_test),
                            shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=20, min_lr=1e-6)
        # tb = TensorBoard(log_dir='./tmp/log', histogram_freq=10)
        filepath = "./NASA_model/{}_classification_dropout_data={}_dropout_net={}_{}".format(model_name,
            dropout_data, dropout_net, idx) + "-improvement.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=True)
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=0,  # epsilon for ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None)#,

        history = model.fit(datagen.flow(x_train, y_train),
                                      # Mygen(x_train, y_train, batch_size=batch_size),
                                      epochs=epochs,
                                      validation_data=(x_test, y_test),
                                      shuffle=True,
                                      steps_per_epoch=int(x_train.shape[0] / batch_size) + 1,
                                      workers=8,
                                      use_multiprocessing=True,
                                      callbacks=[reduce_lr, checkpoint])

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test RMSE:', np.sqrt(scores[0]))
    print('Test MAE:', scores[1])
    model_name = model_name_pre + '{}_dropout_data={}_dropout_net={}_{}'.format(model_name, dropout_data,
                                                                              dropout_net, idx) + 'rotate-RMSE' + str(
        int(np.sqrt(scores[1]) * 100) / 100.0) + '.h5'
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    # plot_history(history, model_path+"-"+str(int(np.sqrt(scores[0])*100)/100.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    plot_history(history, model_path + "-" + str(int(scores[1] * 10) / 10.0) + ".png",
                 100)  # plot_history(history=history, ignore_num=5)
    print("Over!!!")
    return model_path

def evaluation_rotated(model_name, model_path, test_data_path, y_test_path, dropout_data=0, dropout_net=0, idx=0):
    test_data = np.load(test_data_path).astype('float32')

    y_test = np.load(y_test_path).astype('float32')
    y_label = np.zeros(y_test.shape)
    y_label[y_test <= 64] = 0
    y_label[(y_test > 64) & (y_test <= 95)] = 1
    y_label[y_test > 96] = 2

    if model_name.lower() == 'alexnet':
        regressmodel = AlexNet(W_l1RE=1e-4, W_l2RE=0, shape=(65, 65, 2), dropout_net=dropout_net)
    elif model_name.lower() == 'vgg':
        regressmodel = vgg(W_l1RE=1e-4, W_l2RE=0, shape=(65, 65, 2), dropout_net=dropout_net)
    else:
        regressmodel = Network_classification_Dense_dropout(W_l1RE=1e-4, shape=(65, 65, 2), dropout_net=dropout_net)

    regressmodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)
    Rotated_Max_Sita = 45
    y_predict = np.zeros((y_test.shape[0], 8))
    y_class_predict = np.zeros((y_test.shape[0], 8))
    rmses = []
    for rotatedsita in range(0, 360, Rotated_Max_Sita):
        testx = rotate_by_channel(test_data, np.ones(test_data.shape[0])*rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])
        #
        # x_test1 = np.expand_dims(testx[:, :, :, 0], -1)
        # x_test2 = np.expand_dims(testx[:, :, :, 1], -1)

        y_predict_regress = regressmodel.predict(testx, batch_size=128, verbose=0).reshape(-1)
        print("Test data rotated sita: ", rotatedsita)
        y_predict[:, int(rotatedsita/Rotated_Max_Sita)] = y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
        print(str(rotatedsita/Rotated_Max_Sita + 1) + "- rotated blend RMSE: " + str(rmse))
        rmses.append(np.sqrt(np.mean((y_predict_regress[y_label==1]-y_test[y_label==1]) * (y_predict_regress[y_label==1]-y_test[y_label==1]))))
        y_class_predict_tmp = regressmodel.predict(testx, batch_size=128, verbose=0)
        if(len(y_class_predict_tmp.shape)==3):
            y_class_predict_tmp = y_class_predict_tmp.reshape(-1)
        y_class_predict = y_class_predict + y_class_predict_tmp

    y_predict_mean = np.mean(y_predict, axis = -1)
    y_predict_var = np.var(y_predict, axis = -1)
    rmse = np.sqrt(np.mean((y_predict_mean-y_test) * (y_predict_mean-y_test)))
    print("Total - rotated blend RMSE: " + str(rmse))

    prefix = '{}_dropout_data_{}_dropout_net_{}_{}_'.format(model_name, dropout_data, dropout_net, idx)
    var_y = np.reshape(y_predict_var, y_predict_var.shape[0])

    with open("./dropout/TC/" + prefix + "uncertainty_rotate2.txt", "w") as f:
        for idx, v_ in enumerate(var_y):
            f.write('{:.9f}, class {}\n'.format(v_, y_label[idx]))

    dy_list = []
    dy = y_predict_mean - y_test

    with open("./dropout/TC/" + prefix + "dy_rotate2.txt", "w") as f1:
        for d_ in dy:
            dy_list.append(d_)
            f1.write(str(d_) + "\n")

    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)
    y_pre_array = np.array(y_predict_mean)
    #sorted_x_data = test_data[sorted_index,:,:,:]
    sorted_y_pred = y_pre_array[sorted_index]
    sorted_y_data = y_test[sorted_index]
    rmse_list = []

    with open("./dropout/TC/" + prefix + "rmse_rotate2.txt", "w") as f2:
        for i in range(1, total_num + 1):
            sorted_y_pred1 = sorted_y_pred[0:i]
            sorted_y_data1 = sorted_y_data[0:i]
            rmse = np.sqrt(np.mean((sorted_y_pred1 - sorted_y_data1) *
                                   (sorted_y_pred1 - sorted_y_data1)))
            rmse_list.append(rmse)
            f2.write(str(rmse) + "\n")

    sorted_dy = dy[sorted_index]
    epochs = np.arange(total_num)


    plt.figure(0)
    plt.bar(range(0, 360, Rotated_Max_Sita), rmses, width=10)
    plt.xlabel('Rotation Sita',fontsize=16)
    plt.ylabel('RMSE',fontsize=16)
    plt.grid()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+prefix+'RRMSE_rotate2.png',bbox_inches='tight')
    plt.clf()


    plt.figure(1)
    plt.plot(epochs, sorted_dy, 'bo', label='Training loss')  # bo:blue dot蓝点
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('f(x)-y',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+prefix+'dy_rotate2.png',bbox_inches = 'tight')
    plt.clf()

    plt.figure(2)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage',fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+prefix+'rmse_rotate2.png',bbox_inches = 'tight')
    plt.clf()

    plt.figure(3)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage',fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.twinx()
    plt.ylabel('Uncertainty(log)',fontsize=16)
    plt.plot((epochs + 1) / total_num, np.log(var_y[sorted_index] + np.e), 'r-', marker="x")  # bo:blue dot蓝点
    plt.savefig('./dropout/TC/'+prefix+'rmse_uncertainty_rotate2.png',bbox_inches = 'tight')
    plt.clf()

    sorted_index2 = np.argsort(y_predict_var[y_label == 1])
    var_y2 = var_y[y_label == 1]
    plt.figure(4)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage', fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.twinx()
    plt.ylabel('Uncertainty(log)',fontsize=16)
    plt.plot((epochs + 1) / total_num, np.log(var_y[sorted_index] + np.e), 'r-', marker="x")  # bo:blue dot蓝点
    plt.plot((np.arange(len(sorted_index2)) + 1) / len(sorted_index2), np.log(var_y2[sorted_index2] + np.e), 'g-',
             marker="x")  # bo:blue dot蓝点
    plt.legend(['total', 'Cat1+2'])
    plt.savefig('./dropout/TC/' + prefix + 'rmse_uncertainty_case_rotate2.png', bbox_inches='tight')
    plt.clf()

    y_sample = y_test[y_label == 1]
    x_sample = test_data[y_label == 1]
    y_predict_regress = np.zeros((y_sample.shape[0], 60))
    for rotatedsita in tqdm(range(0, 360, 6)):
        testx = rotate_by_channel(x_sample, np.ones(x_sample.shape[0]) * rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])

        y_predict_regress[:, int(rotatedsita/6)] = (regressmodel.predict(testx, verbose=0).reshape(-1))

    y_predict_regress = np.array(y_predict_regress)

    ############ CDF #############
    for i in range(len(y_predict_regress)):
        x_min = np.min(y_predict_regress[i, :])
        x_max = np.max(y_predict_regress[i, :])
        if (y_sample[i] > x_min) and (y_sample[i] < (x_max - 5)):
            plt.figure(5)
            plt.plot([x_min, y_sample[i], y_sample[i], x_max], [0, 0, 1, 1], 'r', label="GT")
            hist, bin_edges = np.histogram(y_predict_regress[i, :], bins=20)
            cdf = np.cumsum(hist) / np.sum(hist)
            cdf2 = np.zeros(cdf.shape[0] + 1)
            cdf2[1:] = cdf
            plt.plot(bin_edges, cdf2, label='CDF')
            plt.xlabel('TC', fontsize=16)
            plt.ylabel('Probability', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend()
            plt.savefig('./dropout/TC/' + prefix + 'CDF_dropout2.png', bbox_inches='tight')
            plt.clf()
            break


    plt.figure(6)
    for i in range(5):
        plt.hlines(y_predict_regress[i, :], i - 0.2, i + 0.2, alpha=0.8, color='g')
    plt.xlabel('Case',fontsize=16)
    plt.ylabel('Prediction',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+prefix+'RRMSE_rotate_case.png', bbox_inches='tight')
    plt.clf()

    plt.figure()
    test_data1 = test_data[y_label == 1]
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_data1[i, 18:83, 18:83, 0], 'gray')
        plt.axis('off')

        plt.subplot(2, 5, i+6)
        plt.imshow(test_data1[i, 18:83, 18:83, 1], 'gray')
        plt.axis('off')

    plt.savefig('./dropout/TC/'+prefix+'RRMSE_image_case.png', dpi=400)

    return y_class_predict, dy

def evaluation(model_name, model_path, test_data_path, y_test_path, times=50, dropout_data=0, dropout_net=0, idx=0):

    test_data = np.load(test_data_path).astype('float32')
    y_test = np.load(y_test_path).astype('float32')
    y_label = np.zeros(y_test.shape)
    y_label[y_test <= 64] = 0
    y_label[(y_test > 64) & (y_test <= 95)] = 1
    y_label[y_test > 96] = 2

    testx = test_data[:, 18:83, 18:83, :]
    testx = normalize_data(testx, testx.shape[3])
    testx1 = testx[:4000]
    testx2 = testx[4000:]
    test_x1 = testx[y_label==1]
    if model_name.lower() == 'alexnet':
        regressmodel = AlexNet(W_l1RE=1e-4, W_l2RE=0, shape=(65, 65, 2), dropout_net=dropout_net)
    elif model_name.lower() == 'vgg':
        regressmodel = vgg(W_l1RE=1e-4, W_l2RE=0, shape=(65, 65, 2), dropout_net=dropout_net)
    else:
        regressmodel = Network_classification_Dense_dropout(W_l1RE=1e-4, shape=(65, 65, 2), dropout_net=dropout_net)
    regressmodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    y_predict = np.zeros((y_test.shape[0], times))

    for rotatedsita in tqdm(range(0, times)):
        # y_predict_regress = []
        # for idx in range(len(testx//128)):
        #     y_predict_regress.extend(np.squeeze(regressmodel.__call__(testx[idx*128: (idx+1)*128], training=True)))
        # y_predict_regress.extend(np.squeeze(regressmodel.__call__(testx[(len(testx//128)):-1], training=True)))
        y_predict_regress1 = np.squeeze(regressmodel.__call__(testx1, training=True))
        y_predict_regress2 = np.squeeze(regressmodel.__call__(testx2, training=True))
        y_predict_regress = []
        y_predict_regress.extend(y_predict_regress1)
        y_predict_regress.extend(y_predict_regress2)
        y_predict[:, rotatedsita] = y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
        #print(str(rotatedsita/Rotated_Max_Sita+1) + "- rotated blend RMSE: " + str(rmse))

    y_predict_mean = np.mean(y_predict, axis = -1)
    y_predict_var = np.var(y_predict, axis =-1)
    rmse = np.sqrt(np.mean((y_predict_mean-y_test) * (y_predict_mean-y_test)))
    print("Total - rotated blend RMSE: " + str(rmse))

    prefix = '{}_dropout_data_{}_dropout_net_{}_{}_'.format(model_name, dropout_data, dropout_net, idx)
    var_y = np.reshape(y_predict_var, y_predict_var.shape[0])

    with open("./dropout/TC/" + prefix + "uncertainty_rotate2.txt", "w") as f:
        for idx, v_ in enumerate(var_y):
            f.write('{:.9f}, class {}\n'.format(v_, y_label[idx]))

    dy_list = []
    dy = y_predict_mean - y_test

    with open("./dropout/TC/" + prefix + "dy_rotate2.txt", "w") as f1:
        for d_ in dy:
            dy_list.append(d_)
            f1.write(str(d_) + "\n")

    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)
    y_pre_array = np.array(y_predict_mean)
    sorted_y_pred = y_pre_array[sorted_index]
    sorted_y_data = y_test[sorted_index]
    rmse_list = []

    with open("./dropout/TC/" + prefix + "rmse_dropout2.txt", "w") as f2:
        for i in range(1, total_num + 1):
            sorted_y_pred1 = sorted_y_pred[0:i]
            sorted_y_data1 = sorted_y_data[0:i]
            rmse = np.sqrt(np.mean((sorted_y_pred1 - sorted_y_data1) *
                                   (sorted_y_pred1 - sorted_y_data1)))
            rmse_list.append(rmse)
            f2.write(str(rmse) + "\n")

    sorted_dy = dy[sorted_index]
    epochs = np.arange(total_num)
    idx = 1234
    x_min = np.min(y_predict[idx, :])
    x_max = np.max(y_predict[idx, :])

    ############# CDF #############
    plt.figure(0)
    plt.plot([x_min, y_test[idx], y_test[idx], x_max], [0, 0, 1, 1], 'r', label="GT")
    hist, bin_edges = np.histogram(y_predict[idx, :], bins=20)
    cdf = np.cumsum(hist) / np.sum(hist)
    cdf2 = np.zeros(cdf.shape[0]+1)
    cdf2[1:] = cdf
    plt.plot(bin_edges, cdf2, label='CDF')
    plt.xlabel('TC', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.savefig('./dropout/TC/' + prefix + 'CDF_dropout2.png', bbox_inches='tight')
    plt.clf()

    plt.figure(1)
    plt.plot(epochs, sorted_dy, 'bo', label='Training loss')  # bo:blue dot蓝点
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('f(x)-y',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+prefix+'dy_dropout2.png',bbox_inches = 'tight')
    plt.clf()
    plt.figure(2)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage',fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+prefix+'rmse_dropout2.png',bbox_inches='tight')
    plt.clf()

    plt.figure(3)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage',fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.twinx()
    plt.ylabel('Uncertainty(log)',fontsize=16)
    plt.plot((epochs + 1) / total_num, np.log(var_y[sorted_index] + np.e), 'r-', marker="x")  # bo:blue dot蓝点
    plt.savefig('./dropout/TC/'+prefix+'rmse_uncertainty_dropout2.png',bbox_inches = 'tight')
    plt.clf()

    sorted_index2 = np.argsort(y_predict_var[y_label == 1])
    plt.figure(4)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage', fontsize=16)
    plt.ylabel('Risk(RMSE)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.twinx()
    plt.ylabel('Uncertainty(log)',fontsize=16)
    plt.plot((epochs + 1) / total_num, np.log(var_y[sorted_index] + np.e), 'r-', marker="x")  # bo:blue dot蓝点
    plt.plot((np.arange(len(sorted_index2)) + 1) / len(sorted_index2), np.log(var_y[y_label == 1][sorted_index2] + np.e), 'g-',
             marker="x")  # bo:blue dot蓝点
    plt.legend(['total', 'Cat1+2'])
    plt.savefig('./dropout/TC/' + prefix + 'rmse_uncertainty_case_dropout2.png', bbox_inches='tight')
    plt.clf()

    plt.figure(5)
    y_predict1 = y_predict[y_label == 1]
    print(y_predict1.shape)
    for i in range(5):
        plt.hlines(y_predict1[i, :], i - 0.2, i + 0.2, alpha=0.8, color='g')
    plt.xlabel('Case',fontsize=16)
    plt.ylabel('Prediction',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+prefix+'RRMSE_dropout_case.png', bbox_inches='tight')
    plt.clf()

    plt.figure()
    test_data1 = test_data[y_label == 1]
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_data1[i, 18:83, 18:83, 0], 'gray')
        plt.axis('off')

        plt.subplot(2, 5, i+6)
        plt.imshow(test_data1[i, 18:83, 18:83, 1], 'gray')
        plt.axis('off')

    plt.savefig('./dropout/TC/'+prefix+'RRMSE_image_case.png', dpi=400)

    return y_predict, y_predict_var, dy

def sample_hist(model_path, test_data_path, dropout_data, dropout_net, test_label_path):
    #classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    test_data = np.load(test_data_path).astype('float32')
    test_label = np.load(test_label_path).astype('float32')
    if dropout_net > 0:
        regressmodel = Network_classification_Dense_dropout(W_l1RE=0, shape=test_data.shape[1:],
                                                            dropout_net=dropout_net)
    else:
        regressmodel = Network_classification(W_l1RE=0, shape=test_data.shape[1:])
    regressmodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    testx = test_data[:, 18:83, 18:83, :]
    testx = normalize_data(testx, testx.shape[3])

    for sample in range(100, 600, 10):
        pred_list = []
        for i in range(100):
            x_sample = np.expand_dims(testx[sample], 0)
            if dropout_data > 0:
                drop_test = np.random.uniform(0, 1, size=x_sample.shape)
                drop_test[drop_test >= dropout_data] = 1
                drop_test[drop_test < dropout_data] = 0
                x_sample = x_sample * drop_test

            y_predict_regress = regressmodel.__call_(x_sample, training=True).reshape(-1)
            pred_list.append(y_predict_regress)

        pred_list = np.array(pred_list).reshape(-1)

        if dropout_data > 0:
            plt.figure()
            plt.hist(pred_list, alpha=0.8)
            plt.savefig('./re_output_0_01_2/dropout_data_{}_hist-{}.png'.format(dropout_data, sample))

            np.save('./re_output_0_01_2/dropout_data_{}_hist-{}.npy'.format(dropout_data, sample), pred_list)
        else:
            plt.figure()
            plt.hist(pred_list, alpha=0.8)
            plt.savefig('./re_output_0_01_2/dropout_net_{}_hist-{}.png'.format(dropout_net, sample))

            np.save('./re_output_0_01_2/dropout_net_{}_hist-{}.npy'.format(dropout_net, sample), pred_list)

        print(np.mean(pred_list), np.std(pred_list), test_label[sample])
    print('done')

def merge_fig(dropout_data, dropout_net):
    for sample in range(100, 600, 10):
        array1 = np.load('./re_output_0_01_2/dropout_data_{}_hist-{}.npy'.format(dropout_data, sample))
        array2 = np.load('./re_output_0_01_2/dropout_net_{}_hist-{}.npy'.format(dropout_net, sample))

        plt.figure(figsize=(10, 7))
        plt.hist(array1, alpha=0.8)
        plt.hist(array2, alpha=0.8)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(['dropout with data', 'dropout with net'], fontsize=16)
        plt.savefig('./re_output_0_01_2/dist_dropout_data_{}_dropout_net_{}_hist-{}.png'.
                    format(dropout_data, dropout_net, sample))

import pandas as pd
def merge_stat(dropout_net):
    df = None
    mean1, mean2, mean3 = [], [], []
    for model in ['alexnet', 'vgg', 'TCIE']:
        for idx in range(0, 4):
            prefix = '{}_dropout_data_{}_dropout_net_{}_{}_'.format(model, 0, dropout_net, idx)
            if dropout_net == 0:
                filename = "./dropout/TC/" + prefix + "uncertainty_rotate2.txt"
            else:
                filename = "./dropout/TC/" + prefix + "uncertainty_rotate2.txt"

            if df is None:
                df = pd.read_csv(filename, delimiter=', ', header=None)
                df.columns = ['uncertainty', 'class']


            else:
                new_df = pd.read_csv(filename, delimiter=', ', header=None)

                df['uncertainty'] = df['uncertainty'] * idx / (idx + 1) + new_df[0] / (idx + 1)
                new_df.columns = ['uncertainty', 'class']

        mean1.append(df[(df['class'] == 'class 0.0')]['uncertainty'].mean())
        mean2.append(df[(df['class'] == 'class 1.0')]['uncertainty'].mean())
        mean3.append(df[(df['class'] == 'class 2.0')]['uncertainty'].mean())

    title = 'MC dropout Uncertainty' if dropout_net > 0 else 'Rotate Uncertainty'
    mean = np.array([mean1, mean2, mean3])
    width = 0.25
    x = np.arange(3)
    plt.figure(figsize=(10, 8))
    plt.ylabel(title, fontsize=20)
    plt.grid()
    plt.bar(x - width, mean[:, 0], width, label='AlexNet')
    plt.bar(x, mean[:, 1], width, label='VGG')
    plt.bar(x + width, mean[:, 2], width, label='TCIE')
    plt.xticks([0, 1, 2], ['AlexNet', 'VGG', 'TCIE'], fontsize=20)
    plt.legend(['TS', 'Cat 1+2', 'Major'], fontsize=20)
    plt.savefig('dropout_net_{}_'.format(dropout_net)+'mean.png', dpi=400)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-DROP1', "--dropout_data", default=0, type=float)
    parser.add_argument('-DROP2', "--dropout_net", default=0.5, type=float)
    parser.add_argument('-I', "--idx", default=0, type=int)
    parser.add_argument('-M', "--model", default='alexnet', type=str)

    parser.add_argument("-P", "--datapath", default="../Data/TCIR-ATLN_EPAC_WPAC.h5", help="the TCIR dataset file path")
    parser.add_argument("-Tx", "--trainset_xpath", default="./Data/ATLN_2003_2014_data_x_101.npy", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="./Data/ATLN_2003_2014_data_y_101.npy", help="the trainning set y file path")

    parser.add_argument("-Tex", "--testset_xpath", default="./Data/ATLN_2015_2016_data_x_101.npy", help="the test set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="./Data/ATLN_2015_2016_data_y_101.npy", help="the test set y file path")

    parser.add_argument("-E", "--epoch", default=300, type=int, help="epochs for trainning")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sita", default=50, type=int, help="rotated sita for blending")
    args = parser.parse_args()
    # model = siamese_net((60, 60, 1), dropout=0)
    # plot_model(model, show_shapes=True)
    merge_stat(args.dropout_net)
    # if (args.dropout_data > 0) and (args.dropout_net > 0):
    #     merge_fig(dropout_data=args.dropout_data, dropout_net=args.dropout_net)
    #     sys.exit()

    # if args.test:
    #     model_path = ''
    #     for item in os.listdir('./result_model'):
    #         if ('{}_dropout_data={}_dropout_net={}_{}rotate'.format(args.model, args.dropout_data,
    #     args.dropout_net, int(args.idx)) in item) and ('png' not in item) and ('RMSE' in item):
    #             print(item)
    #             model_path = os.path.join('./result_model/', item)
    #             break
    #
    #     if args.dropout_net > 0:
    #         evaluation(args.model, model_path, args.testset_xpath, args.testset_ypath, times=args.sita,
    #                        dropout_data=args.dropout_data, dropout_net=args.dropout_net, idx=args.idx)
    #     else:
    #         evaluation_rotated(args.model, model_path, args.testset_xpath, args.testset_ypath,
    #                        dropout_data=args.dropout_data, dropout_net=args.dropout_net, idx=args.idx)
    # else:
    #     model_path = train_net(args.model, args.epoch, args.trainset_xpath, args.trainset_ypath, args.testset_xpath,
    #                                args.testset_ypath, args.dropout_data, args.dropout_net, args.idx)

        # if args.dropout_net > 0:
        #     evaluation(args.model, model_path, args.testset_xpath, args.testset_ypath, times=args.sita,
        #                    dropout_data=args.dropout_data, dropout_net=args.dropout_net, idx=args.idx)
        # else:
        #     evaluation_rotated(args.model, model_path, args.testset_xpath, args.testset_ypath,
        #                    dropout_data=args.dropout_data, dropout_net=args.dropout_net, idx=args.idx)
        # sample_hist(model_path, args.testset_xpath, dropout_data=args.dropout_data, dropout_net=args.dropout_net,
        #             test_label_path=args.testset_ypath)
