import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
pylab.rcParams['figure.figsize'] = (32.0, 24.0)
pylab.rcParams['font.size'] = 24

import mnist_data as mnist_data
import vae as vae
from helpers import batch_index_groups, dtype

train_total_data, _, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()

train_size = 10000
IMAGE_SIZE_MNIST = 28
num_hidden = 500
dim_img = IMAGE_SIZE_MNIST ** 2
dim_z = 2
learn_rate = 1e-3
batch_size = min(128, train_size)
num_epochs = 101

y_input = tf.placeholder(dtype, shape=[None, dim_img], name='input_img')
y_output_true = tf.placeholder(dtype, shape=[None, dim_img], name='target_img')

# dropout
keep_prob = tf.placeholder(dtype, name='keep_prob')

z = tf.placeholder(dtype, name='z')

# network architecture
ae = vae.autoencoder(
    y_input=y_input,
    y_output_true=y_output_true,
    dim_img=dim_img,
    dim_z=dim_z,
    num_hidden=num_hidden,
    keep_prob=keep_prob
)

# optimization
train_step = tf.train.AdamOptimizer(learn_rate).minimize(ae.loss)

y_train = train_total_data[:train_size, :-mnist_data.NUM_LABELS]
y_train_labels = train_total_data[:train_size, -mnist_data.NUM_LABELS:]

epsilon = 1e-6
img = vae.bernoulli_mlp_decoder(z, num_hidden, dim_img, 0.9, reuse=True)
img = tf.clip_by_value(
    t=img,
    clip_value_min=epsilon,
    clip_value_max=1 - epsilon
)

print("Num data points", train_size)
print("Num epochs", num_epochs)

with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    session.graph.finalize()

    for epoch in range(num_epochs):
        for i, batch_indices in enumerate(batch_index_groups(batch_size=batch_size, num_samples=train_size)):

            batch_xs_input = y_train[batch_indices, :]

            _, tot_loss, loss_likelihood, loss_divergence, aez = session.run(
                (
                    train_step,
                    ae.loss,
                    ae.neg_marginal_likelihood,
                    ae.kl_divergence,
                    ae.z
                ),
                feed_dict={
                    y_input: batch_xs_input,
                    y_output_true: batch_xs_input,
                    keep_prob: 0.9
                }
            )

        print(
            "epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch,
                tot_loss,
                loss_likelihood,
                loss_divergence
            )
        )

        if epoch % 10 == 0:
            imgOut = session.run(
                img,
                feed_dict={
                    z: aez,
                }
            )

            imgOut = imgOut.reshape(16, 8, 28, 28).swapaxes(1, 2).reshape(16*28, -1)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            plt.imshow(imgOut, cmap='gray')
            plt.savefig('plt' + str(int(epoch/10)) + '.png')
