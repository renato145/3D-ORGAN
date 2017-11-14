from .layers import dense_layer, conv_layer
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Activation
from keras.layers import Embedding, Lambda, Concatenate, Add
from keras.layers.convolutional import Conv3DTranspose, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K

# Discriminator models:

def _discriminator(dict_size, se=False):
    # inputs
    labels = Input(shape=(1,))
    voxels_inp = Input(shape=(32,32,32))
    voxels = Lambda(lambda x: K.expand_dims(x))(voxels_inp)
    
    # label embedding
    embs = Embedding(dict_size, 64, input_length=1)(labels)
    embs = Flatten()(embs)
    embs = dense_layer(embs, 1024, act='lrelu', bn=False)
    
    # conv layers
    out = conv_layer(voxels, 32, 5, 1, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 32, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 64, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 128, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 256, act='lrelu', bn=False, se=se)
    out = Flatten()(out)
    out = dense_layer(out, 1024, act='lrelu', bn=False)
    out = Concatenate()([out, embs])
    out = dense_layer(out, 1024, act='lrelu', bn=False)
    out = dense_layer(out, 512, act='lrelu', bn=False)
    out = dense_layer(out, 1, act=None, bn=False)
    
    return Model((voxels_inp, labels), out)

# -----------------------------------------------------------------------
# Generator models:

def _generator_v(dict_size):
    # inputs
    labels = Input(shape=(1,))
    voxels_inp = Input(shape=(32,32,32))
    voxels = Lambda(lambda x: K.expand_dims(x))(voxels_inp)
    
    # label embedding
    embs = Embedding(dict_size, 64, input_length=1)(labels)
    embs = Flatten()(embs)
    embs = dense_layer(embs, 1024)
    
    # conv layers
    out = conv_layer(voxels, 32, 5, 1)
    out = conv_layer(out, 32)
    out = conv_layer(out, 64)
    out = conv_layer(out, 128)
    out = conv_layer(out, 256)
    
    out = Flatten()(out)
    out = dense_layer(out, 1024)
    out = Concatenate()([out, embs])
    out = dense_layer(out, 1024) # as in D
    out = dense_layer(out, 2*2*2*256)
    out = Lambda(lambda x: K.reshape(x, (-1,2,2,2,256)))(out)

    out = conv_layer(out, 256, transpose=True)
    out = conv_layer(out, 128, transpose=True)
    out = conv_layer(out, 64, transpose=True)
    out = conv_layer(out, 32, transpose=True)

    out = conv_layer(out, 1, 5, 1, act='tanh', bn=False)
    out = Lambda(lambda x: K.squeeze(x, 4))(out)
    
    return Model((voxels_inp, labels), out)

def _generator_u(dict_size, se=False):
    # inputs
    labels = Input(shape=(1,))
    voxels_inp = Input(shape=(32,32,32))
    voxels = Lambda(lambda x: K.expand_dims(x))(voxels_inp)
    
    # label embedding
    embs = Embedding(dict_size, 64, input_length=1)(labels)
    embs = Flatten()(embs)
    embs = dense_layer(embs, 1024)
    
    # conv layers
    encoder1 = conv_layer(voxels, 32, 5, 1, se=se)
    encoder2 = conv_layer(encoder1, 32, se=se)
    encoder3 = conv_layer(encoder2, 64, se=se)
    encoder4 = conv_layer(encoder3, 128, se=se)
    encoder5 = conv_layer(encoder4, 256, se=se)
    
    mix = Flatten()(encoder5)
    mix = dense_layer(mix, 1024)
    mix = Concatenate()([mix, embs])
    mix = dense_layer(mix, 1024) # as in D
    mix = dense_layer(mix, 2*2*2*256)
    mix = Lambda(lambda x: K.reshape(x, (-1,2,2,2,256)))(mix)
    mix = Concatenate()([mix, encoder5])

    decoder1 = conv_layer(mix, 128, transpose=True, se=se)
    decoder1 = Concatenate()([decoder1, encoder4])
    decoder2 = conv_layer(decoder1, 64, transpose=True, se=se)
    decoder2 = Concatenate()([decoder2, encoder3])
    decoder3 = conv_layer(decoder2, 32, transpose=True, se=se)
    decoder3 = Concatenate()([decoder3, encoder2])
    decoder4 = conv_layer(decoder3, 32, transpose=True, se=se)
    decoder4 = Concatenate()([decoder4, encoder1])

    out = conv_layer(decoder4, 1, 5, 1, act='tanh', bn=False)
    out = Lambda(lambda x: K.squeeze(x, 4))(out)
    
    return Model((voxels_inp, labels), out)

# -----------------------------------------------------------------------
# Wrappers:

def make_discriminator(dict_size, model_type):
    model = {
        'voxels-v': _discriminator(dict_size),
        'voxels-u': _discriminator(dict_size),
        'voxels-use': _discriminator(dict_size, se=True)
    }
    
    return model[model_type]

def make_generator(dict_size, model_type):
    model = {
        'voxels-v': _generator_v(dict_size),
        'voxels-u': _generator_u(dict_size),
        'voxels-use': _generator_u(dict_size, se=True)
    }
    
    return model[model_type]