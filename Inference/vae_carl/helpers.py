
import tensorflow as tf
import numpy as np

dtype = tf.float64


def batch_index_groups(batch_size, num_samples):

    indices = np.random.choice(num_samples, num_samples, replace=False)

    def offset(i): return (i * batch_size) % num_samples

    return [
        indices[offset(i):(offset(i) + batch_size)]
        for i in range(int(num_samples / batch_size))
    ]


def kl_divergence_univariate_gaussians(q_mu, q_sigma, p_mu, p_sigma):
    epsilon = 1e-8
    q_sigma_sqrd = tf.square(q_sigma)
    p_sigma_sqrd = tf.square(p_sigma)
    t0 = tf.log(epsilon + (p_sigma_sqrd / q_sigma_sqrd))
    t1 = (q_sigma_sqrd + tf.square(q_mu - p_mu)) / p_sigma_sqrd
    t2 = -1
    return tf.reduce_mean(tf.reduce_sum(0.5 * (t0 + t1 + t2), axis=1))


def gaussian_sample(mu, sigma):
    return mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=dtype)