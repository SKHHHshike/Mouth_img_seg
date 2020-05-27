from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder
from .lw_rf_net import get_lw_rf_net_encoder


if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def unet_mini(n_classes, input_height=360, input_width=480):
    # pretrained_weights = None
    if IMAGE_ORDERING == 'channels_first':
        inputs = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        inputs = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format=IMAGE_ORDERING)(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format=IMAGE_ORDERING)(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format=IMAGE_ORDERING)(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv4)
    drop4 = Dropout(0.2)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(drop5))
    # merge6 = concatenate([drop4, up6], axis=3)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(
        UpSampling2D(size=(2, 2), data_format=IMAGE_ORDERING)(drop4))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(
        UpSampling2D(size=(2, 2), data_format=IMAGE_ORDERING)(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(
        UpSampling2D(size=(2, 2), data_format=IMAGE_ORDERING)(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv9)
    conv9 = Conv2D(n_classes, 1, padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING,activation='sigmoid')(conv9)
    # conv10 = Conv2D(n_classes, 1, activation='sigmoid', data_format=IMAGE_ORDERING)(conv9)

    # o = Model(input=inputs, output=conv10)
    model = get_segmentation_model(inputs, conv9)
    # model.model_name = "unet_mini"
    model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss='dice_coef', metrics=['accuracy'])
    #
    # # model.summary()
    #
    # if (pretrained_weights):
    #     model.load_weights(pretrained_weights)
    #
    #
    # model.output_width = input_width
    # model.output_height = input_height
    # model.n_classes = n_classes
    # model.input_height = input_height
    # model.input_width = input_width
    model.model_name = "unet_mini"

    return model
    # if IMAGE_ORDERING == 'channels_first':
    #     img_input = Input(shape=(3, input_height, input_width))
    # elif IMAGE_ORDERING == 'channels_last':
    #     img_input = Input(shape=(input_height, input_width, 3))
    #
    # conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(img_input)
    # conv1 = Dropout(0.2)(conv1)
    # conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(conv1)
    # pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)
    #
    # conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(pool1)
    # conv2 = Dropout(0.2)(conv2)
    # conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(conv2)
    # pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)
    #
    # conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(pool2)
    # conv3 = Dropout(0.2)(conv3)
    # conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(conv3)
    #
    # up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
    #     conv3), conv2], axis=MERGE_AXIS)
    # conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(up1)
    # conv4 = Dropout(0.2)(conv4)
    # conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(conv4)
    #
    # up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
    #     conv4), conv1], axis=MERGE_AXIS)
    # conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(up2)
    # conv5 = Dropout(0.2)(conv5)
    # conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
    #                activation='relu', padding='same')(conv5)
    #
    # o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
    #            padding='same')(conv5)
    #
    # model = get_segmentation_model(img_input, o)
    # model.model_name = "unet_mini"
    # return model




def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)

    return model



def unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "unet"
    return model


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "vgg_unet"
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "resnet50_unet"
    return model

def lw_rf_net_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3):

    model = _unet(n_classes, get_lw_rf_net_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "lw_rf_net_unet"
    return model


def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "mobilenet_unet"
    return model


if __name__ == '__main__':
    # m = unet_mini(101)
    # m = _unet(101, vanilla_encoder)
    # # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    # m = _unet(101, get_vgg_encoder)
    # m = _unet(101, get_resnet50_encoder)
    model = unet_mini(2,128,256)
    param = model.summary()
    print("param = " + str(param))
