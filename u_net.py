from keras.models import Model
from keras import backend as K
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization,\
    Conv2DTranspose, Add, Reshape, Lambda
from keras.optimizers import RMSprop
from losses import bce_dice_loss, dice_loss
import tensorflow as tf
from keras.metrics import Accuracy, MeanSquaredError, AUC, Precision, Recall, MeanIoU


def overlay(x):
    np_mask = x[0].numpy()
    np_input = x[1].numpy()

    np_input[np_mask == 0] = 0

    tensor_overlay = K.constant(np_input)
    return tensor_overlay


def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    input_size = input_shape[0]
    nClasses = 9  # 9 keypoints
    input_height, input_width, sigma = 128, 128, 5

    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(32, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(32, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(64, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(64, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(128, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(128, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(256, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(256, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(512, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(512, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    output1 = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    lamb = Lambda(overlay)([output1, inputs])

    k1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format="channels_last")(lamb)
    k1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format="channels_last")(k1)
    block1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format="channels_last")(k1)

    # Encoder Block 2
    k2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format="channels_last")(block1)
    k2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format="channels_last")(k2)
    k2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format="channels_last")(k2)

    # bottoleneck
    k3 = (Conv2D(32 * 5, (int(input_height / 4), int(input_width / 4)), activation='relu', padding='same',
                 name="bottleneck_1", data_format="channels_last"))(k2)
    k3 = (Conv2D(32 * 5, (1, 1), activation='relu', padding='same', name="bottleneck_2", data_format="channels_last"))(
        k3)

    # upsamping to bring the feature map size to be the same as the one from block1
    o_block1 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name='upsample_1',
                         data_format="channels_last")(k3)
    o = Add()([o_block1, block1])
    output2 = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name='upsample_2',
                        data_format="channels_last")(o)

    model = Model(inputs=inputs, outputs=[output1, output2])
    model.summary()

    # model.compile(optimizer=RMSprop(lr=0.001), loss=bce_dice_loss, metrics=[dice_loss])
    # model.compile(loss='mse', optimizer=RMSprop(lr=0.001), sample_weight_mode="temporal",
    #               metrics=[])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss={'output1': bce_dice_loss, 'output2': 'mse'},
                  sample_weight_mode={'output1': None, 'output2': 'temporal'})
                  # metrics={'output1': [dice_loss, MeanIoU(2)], 'output2': [Accuracy(), MeanSquaredError(),
                  #                                                          AUC()]})

    return input_size, model
