from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

from .config import IMAGE_ORDERING
from .resnet50 import get_resnet50_encoder
from .model_utils import get_segmentation_model


if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def refinenet_mini(n_classes, input_height=360, input_width=480):
    if IMAGE_ORDERING == 'channels_first':
        inputs = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        inputs = Input(shape=(input_height, input_width, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv1)
    f1 = conv1
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format=IMAGE_ORDERING)(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv2)
    f2 = conv2
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format=IMAGE_ORDERING)(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv3)
    f3 = conv3
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format=IMAGE_ORDERING)(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(conv4)
    drop4 = Dropout(0.2)(conv4)
    f4 = drop4
    o = f4
    o = refinenet_block_input1(256,o)
    o = refinenet_block_input2(128,o,f3)
    o = refinenet_block_input2(64,o,f2)
    o = refinenet_block_input2(32,o,f1)
    o = Conv2D(n_classes, 1, padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING,
                   activation='sigmoid')(o)

    model = get_segmentation_model(inputs, o)
    model.model_name = "refinenet_mini"
    model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def _refinenet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f4
    o = refinenet_block_input1(256,o)
    o = refinenet_block_input2(128,o,f3)
    o = refinenet_block_input2(64,o,f2)
    o = refinenet_block_input2(32,o,f1)


    # o = Conv2D(n_classes, (3, 3), padding='same',
    #            data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)

    return model

def refinenet_block_input1(filter,x):
    x1 = residual_conv_unit(filter,x)
    x2 = residual_conv_unit(filter,x)
    x = fusion1(filter,x1,x2)
    x = chained_residual_pooling(filter,x)
    x = residual_conv_unit(filter,x)
    return x

def refinenet_block_input2(filter,x,f):
    x1 = residual_conv_unit(filter,x)
    x2 = residual_conv_unit(filter,x)
    f1 = residual_conv_unit(filter,f)
    f2 = residual_conv_unit(filter,f)
    sum = fusion2(filter,x1,x2,f1,f2)
    sum = chained_residual_pooling(filter,sum)
    sum = residual_conv_unit(filter,sum)
    return sum

def residual_conv_unit(filter,x):
    out1 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(x)
    out2 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(out1)
    return concatenate([x, out2], axis=MERGE_AXIS)

def fusion1(filter,x1,x2):
    x1 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(x1)
    x2 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(x2)
    return concatenate([x1, x2], axis=MERGE_AXIS)

def fusion2(filter,x1,x2,f1,f2):

    f1 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(f1)
    f2 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(f2)
    f = (concatenate([f1, f2], axis=MERGE_AXIS))

    pad_f = f.get_shape()[1].value
    pad_f2 = f.get_shape()[2].value
    x1 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(x1)
    pad_x = x1.get_shape()[1].value
    pad_x2 = x1.get_shape()[2].value
    pad = (int)(pad_f/pad_x)
    pad2 =(int)(pad_f2/pad_x2)
    x1 = UpSampling2D(size=(pad, pad2), data_format=IMAGE_ORDERING)(x1)

    x2 = Conv2D(filter, 3, activation='relu', padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(x2)
    x2 = UpSampling2D(size=(pad, pad2), data_format=IMAGE_ORDERING)(x2)
    x =(concatenate([x1, x2], axis=MERGE_AXIS))

    return concatenate([x, f], axis=MERGE_AXIS)
def chained_residual_pooling(filter,x):

    x = Activation('relu')(x)
    sum0 = x
    # sum1 = MaxPooling2D(pool_size=(5, 5), data_format=IMAGE_ORDERING,padding='same')(sum0)
    sum1 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(sum0)

    # sum2 = MaxPooling2D(pool_size=(5, 5), data_format=IMAGE_ORDERING,padding='same')(sum1)
    sum2 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(sum1)

    # sum3 = MaxPooling2D(pool_size=(5, 5), data_format=IMAGE_ORDERING,padding='same')(sum2)
    sum3 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(sum2)

    sum = (concatenate([sum0, sum1], axis=MERGE_AXIS))
    sum = (concatenate([sum, sum2], axis=MERGE_AXIS))
    sum = (concatenate([sum, sum3], axis=MERGE_AXIS))
    # sum = sum0 + sum1 + sum2 + sum3
    return sum


def refinenet_resnet50(n_classes, input_height=416, input_width=608,
                  encoder_level=3):

    model = _refinenet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "refinenet_resnet50"
    return model



if __name__ == '__main__':
    # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    m = _refinenet(101, get_resnet50_encoder)
