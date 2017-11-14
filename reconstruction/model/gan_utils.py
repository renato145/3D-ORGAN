import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from functools import partial

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

def build_gan_arch(discriminator, generator, batch_size, gradient_penalty, loss, loss_multiply, size=32):

    class RandomWeightedAverage(_Merge):
        def _merge_function(self, inputs):
            weights = K.random_uniform((batch_size, 1, 1, 1))
            return (weights * inputs[0]) + ((1 - weights) * inputs[1])

    loss = 'mae' if loss == 'l1' else 'mse'

    for layer in discriminator.layers:
        layer.trainable = False

    discriminator.trainable = False

    generator_input_voxels = Input(shape=(size,size,size))
    generator_input_labels = Input(shape=(1,))

    generator_layers = generator([generator_input_voxels, generator_input_labels])
    discriminator_layers_for_g = discriminator([generator_layers, generator_input_labels])
    generator_model = Model(inputs=[generator_input_voxels, generator_input_labels],
                            outputs=[discriminator_layers_for_g, generator_layers])

    generator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss, loss], loss_weights=[1, loss_multiply])
    # --End Generator Model--
    
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples_voxels = Input(shape=(size,size,size))
    real_samples_labels = Input(shape=(1,))
    generator_input_voxels_for_discriminator = Input(shape=(size,size,size))
    generated_samples_for_discriminator = generator([generator_input_voxels_for_discriminator, real_samples_labels])
    discriminator_output_from_generator = discriminator([generated_samples_for_discriminator, real_samples_labels])
    discriminator_output_from_real_samples = discriminator([real_samples_voxels, real_samples_labels])

    averaged_samples = RandomWeightedAverage()([real_samples_voxels, generated_samples_for_discriminator])
    averaged_samples_out = discriminator([averaged_samples, real_samples_labels])

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penalty)
    partial_gp_loss.__name__ = 'gradient_penalty'
    
    discriminator_model = Model(inputs=[real_samples_voxels, real_samples_labels, generator_input_voxels_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])

    discriminator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    # --End Discriminator Model--
    
    return discriminator_model, generator_model