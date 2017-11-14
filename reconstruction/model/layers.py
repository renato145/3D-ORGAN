from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Activation
from keras.layers import Embedding, Lambda, Concatenate, Add
from keras.layers import GlobalAvgPool3D, Multiply
from keras.layers.convolutional import Conv3DTranspose, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K

def dense_layer(inp, f, act='relu', bn=True):
    initializer = act if act is not None else ''
    initializer = 'he_uniform' if initializer.find('relu') != -1 else 'glorot_uniform'
    out = Dense(f, use_bias=False, kernel_initializer=initializer)(inp)
    if bn: out = BatchNormalization()(out)
    
    if act == 'lrelu':
        out = LeakyReLU(alpha=0.2)(out)
    elif act is not None:
        out = Activation(act)(out)
    
    return out

def conv_layer(inp, f, k=4, s=2, p='same', act='relu', bn=True, transpose=False,
               se=False, se_ratio=16):
    initializer = act if act is not None else ''
    initializer = 'he_uniform' if initializer.find('relu') != -1 else 'glorot_uniform'
    fun = Conv3DTranspose if transpose else Conv3D
    out = fun(f, k, strides=s, padding=p, use_bias=False, kernel_initializer=initializer)(inp)
    if bn: out = BatchNormalization()(out)
    
    if act == 'lrelu':
        out = LeakyReLU(alpha=0.2)(out)
    elif act is not None:
        out = Activation(act)(out)

    # squeeze and excite
    if se:
        out_se = GlobalAvgPool3D()(out)
        r = f // se_ratio if (f // se_ratio) > 0 else 1
        out_se = Reshape((1, 1, f))(out_se)
        out_se = Dense(r, use_bias=False, kernel_initializer='he_uniform',
                       activation='relu')(out_se)
        out_se = Dense(f, use_bias=False, activation='sigmoid')(out_se)
        out = Multiply()([out, out_se])
    
    return out

