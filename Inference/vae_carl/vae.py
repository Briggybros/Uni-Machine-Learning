import tensorflow as tf

from abc import ABCMeta
from helpers import dtype, kl_divergence_univariate_gaussians, gaussian_sample


def autoencoder(
    y_input,
    y_output_true,
    dim_img,
    dim_z,
    num_hidden,
    keep_prob
):
    epsilon = 1e-6

    # encoding
    mu, sigma = gaussian_mlp_encoder(y_input, num_hidden, dim_z, keep_prob, scope_name="z_encoder")

    # unit gaussian
    kl = kl_divergence_univariate_gaussians(
        q_mu=mu,
        q_sigma=sigma,
        p_mu=tf.constant(0, dtype=dtype),
        p_sigma=tf.constant(1, dtype=dtype)
    )

    # sampling by re-parameterization technique
    z = gaussian_sample(mu=mu, sigma=sigma)

    # decoding
    y_output = tf.clip_by_value(
        t=bernoulli_mlp_decoder(z, num_hidden, dim_img, keep_prob),
        clip_value_min=epsilon,
        clip_value_max=1 - epsilon
    )

    marginal_likelihood = tf.reduce_mean(
        tf.reduce_sum(
            y_output_true * tf.log(y_output, name="log_y") +
            (1 - y_output_true) * tf.log(1 - y_output, name="log_1-y"),
            axis=1
        )
    )

    elbo = marginal_likelihood - kl

    neg_marginal_likelihood = -marginal_likelihood
    loss = -elbo

    class VAE(metaclass=ABCMeta):

        @property
        def y_output(self): return y_output

        @property
        def z(self): return z

        @property
        def mu(self): return mu

        @property
        def sigma(self): return sigma

        @property
        def loss(self): return loss

        @property
        def neg_marginal_likelihood(self): return neg_marginal_likelihood

        @property
        def kl_divergence(self): return kl

    return VAE()


# Gaussian MLP as encoder
def gaussian_mlp_encoder(x, n_hidden, n_output, keep_prob, scope_name):
    with tf.variable_scope(scope_name):

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0., dtype=dtype)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init, dtype=dtype)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init, dtype=dtype)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init, dtype=dtype)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init, dtype=dtype)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init, dtype=dtype)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init, dtype=dtype)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

    return mean, stddev


# Bernoulli MLP as decoder
def bernoulli_mlp_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0., dtype=dtype)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.shape[1], n_hidden], initializer=w_init, dtype=dtype)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init, dtype=dtype)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init, dtype=dtype)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init, dtype=dtype)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init, dtype=dtype)
        bo = tf.get_variable('bo', [n_output], initializer=b_init, dtype=dtype)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y