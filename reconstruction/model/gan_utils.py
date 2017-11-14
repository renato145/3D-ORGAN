import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from functools import partial

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

def build_gan_arch(discriminator, generator, batch_size, gradient_penalty, loss, loss_multiply):
    """WGAN-GP"""
    class RandomWeightedAverage(_Merge):
        """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
        between each pair of input points.
        Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
        Improvements appreciated."""

        def _merge_function(self, inputs):
            weights = K.random_uniform((batch_size, 1, 1, 1))
            return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    # Getting loss function
    loss = 'mae' if loss == 'l1' else 'mse'
    # The generator_model is used when we want to train the generator layers.
    # As such, we ensure that the discriminator layers are not trainable.
    # Note that once we compile this model, updating .trainable will have no effect within it. As such, it
    # won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
    # as we compile the generator_model first.
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input_voxels = Input(shape=(32,32,32))
    generator_input_labels = Input(shape=(1,))
    generator_layers = generator([generator_input_voxels, generator_input_labels])
    discriminator_layers_for_g = discriminator([generator_layers, generator_input_labels])
    generator_model = Model(inputs=[generator_input_voxels, generator_input_labels],
                            outputs=[discriminator_layers_for_g, generator_layers])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss, loss], loss_weights=[1, loss_multiply])
    # --End Generator Model--
    
    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples_voxels = Input(shape=(32,32,32))
    real_samples_labels = Input(shape=(1,))
    generator_input_voxels_for_discriminator = Input(shape=(32,32,32))
    generated_samples_for_discriminator = generator([generator_input_voxels_for_discriminator, real_samples_labels])
    discriminator_output_from_generator = discriminator([generated_samples_for_discriminator, real_samples_labels])
    discriminator_output_from_real_samples = discriminator([real_samples_voxels, real_samples_labels])

    # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()([real_samples_voxels, generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never really use the discriminator
    # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator([averaged_samples, real_samples_labels])

    # The gradient penalty loss function requires the input averaged samples to get gradients. However,
    # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
    # of the function with the averaged samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penalty)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error
    
    # Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
    # real samples and generated samples before passing them to the discriminator: If we had, it would create an
    # output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.

    # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
    # samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples_voxels, real_samples_labels, generator_input_voxels_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
    # samples, and the gradient penalty loss for the averaged samples.
    discriminator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    # --End Discriminator Model--
    
    return discriminator_model, generator_model