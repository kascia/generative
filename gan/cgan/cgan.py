from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.framework import get_or_create_global_step

mnist = input_data.read_data_sets('../../mnist/', one_hot = True)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
layers = tf.contrib.layers
slim = tf.contrib.slim


batch_size = 60
noise_size = 100
condition_size = 10
data_size = 28*28
learning_rate = 1e-4
dropout_rate = 1.0
epsilon = 0
tb_dir = '/data/tensorboard_log/leewk92/gan/cgan/'
checkpoint_dir = './checkpoints/'

def generator(z, y):
    with tf.variable_scope('generator'):
        with tf.variable_scope('concatenate'):
            c = tf.concat([z, y], axis=1)
        with slim.arg_scope([layers.fully_connected],
                            activation_fn = tf.nn.relu,
                            weights_initializer = \
                            layers.xavier_initializer(),
                            biases_initializer = \
                            tf.constant_initializer(0.1),
                            weights_regularizer =
                            layers.l2_regularizer(1.)):
            with tf.variable_scope('fully'):
                fc1 = layers.fully_connected(c, 100, scope='fully1')
                fc1 = tf.nn.dropout(fc1, dropout_rate)
                fc2 = layers.fully_connected(fc1, 500, scope='fully2')
                fc2 = tf.nn.dropout(fc2, dropout_rate)

            with tf.variable_scope('generated'):
                generated = layers.fully_connected(fc2, data_size, scope='generated',
                                                activation_fn = tf.nn.tanh)
    return generated

def discriminator(x, y):
    with tf.variable_scope('discriminator'):
        with tf.variable_scope('concatenate'):
            c = tf.concat([x, y], axis=1)
        with slim.arg_scope([layers.fully_connected],
                            activation_fn = tf.nn.relu,
                            weights_initializer = \
                            layers.xavier_initializer(),
                            biases_initializer = \
                            tf.constant_initializer(0.01),
                            weights_regularizer =
                            layers.l2_regularizer(1.)):
            with tf.variable_scope('fully'):
                fc1 = layers.fully_connected(c, 500, scope='fully1')
                fc1 = tf.nn.dropout(fc1, dropout_rate)
                fc2 = layers.fully_connected(fc1, 100, scope='fully2')
                fc2 = tf.nn.dropout(fc2, dropout_rate)

            with tf.variable_scope('score'):
                score = layers.fully_connected(fc2, 1, scope='score',
                                                activation_fn = tf.nn.sigmoid)
    return score


def get_next_batch(batch_size):
    samples, c = mnist.train.next_batch(batch_size)
    samples = np.array(samples, dtype=np.float32)
    samples /= 0.5
    samples -= 1.0
    return samples, c

def generate_random_normal_vector():
    noise = np.random.randn(batch_size, noise_size)
    return noise

def generate_ordered_conditions():
    c = np.zeros([batch_size, condition_size])
    for i in range(batch_size):
        c[i][i] = 1
    return c

def main():

    global_step = get_or_create_global_step()
    z = tf.placeholder(tf.float32, [batch_size, noise_size])
    x = tf.placeholder(tf.float32, [batch_size, data_size])
    y = tf.placeholder(tf.float32, [batch_size, condition_size])

    G = generator(z, y)
    D_real = discriminator(x, y)
    D_fake = discriminator(G, y)
    var_generator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='generator')
    var_discriminator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='discriminator')

    loss_discriminator = tf.reduce_mean(-tf.log(D_real + epsilon) - tf.log(1-D_fake + epsilon),
                                         axis=0)
    #loss_generator = tf.reduce_mean(tf.log(1-D_fake), axis=0)
    loss_generator = tf.reduce_mean(-tf.log(D_fake), axis=0)
    """
    optimize_discriminator = tf.train.AdamOptimizer(learning_rate).minimize(
        loss=loss_discriminator, var_list=var_discriminator)
    optimize_generator = tf.train.AdamOptimizer(learning_rate).minimize(
        loss=loss_generator, var_list=var_generator)
    """
    optimizer_discriminator = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars_d = optimizer_discriminator.compute_gradients(
        loss=loss_discriminator, var_list=var_discriminator)
    clipped_grads_and_vars_d = [(tf.clip_by_norm(grad, 5.0),var) for grad, var
                              in grads_and_vars_d]
    optimize_discriminator = optimizer_discriminator.apply_gradients(
                                clipped_grads_and_vars_d,
                                global_step=global_step)

    optimizer_generator= tf.train.AdamOptimizer(learning_rate)
    grads_and_vars_g = optimizer_generator.compute_gradients(
        loss=loss_generator, var_list=var_generator)
    clipped_grads_and_vars_g = [(tf.clip_by_norm(grad, 5.0),var) for grad, var
                                in grads_and_vars_g]
    optimize_generator = optimizer_generator.apply_gradients(
                            clipped_grads_and_vars_g,
                            global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    print('g', [item.name for item in var_generator])
    print('d', [item.name for item in var_discriminator])

    saver = tf.train.Saver()
    sess = tf.Session()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('load_model', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('initialize_model')
        sess.run(init_op)

    writer = tf.summary.FileWriter(tb_dir, sess.graph)
    for i in range(20000):
        for k in range(1):
            real, condition = get_next_batch(batch_size)
            noise = generate_random_normal_vector()
            """
            _D_real, _D_fake, _loss_d, _loss_g, _, _, m = sess.run(
                                            [D_real,
                                            D_fake,
                                            loss_discriminator,
                                            loss_generator,
                                            optimize_discriminator,
                                            optimize_generator,
                                            merged],
                                            feed_dict={x:real, z:noise})
            """
            _D_real,_loss_d, _ = sess.run([D_real,loss_discriminator,
                                            optimize_discriminator
                                            ],
                                          feed_dict={x:real, y:condition,
                                                     z:noise})

        noise = generate_random_normal_vector()
        real, condition = get_next_batch(batch_size)
        _D_fake,_loss_g, _ = sess.run([D_fake,loss_generator,
                                       optimize_generator
                                       ],
                                      feed_dict={z:noise, y:condition})
        if i % 100 == 0:
            print('%g th step' % i)
            print('D_real : %g' % np.mean(_D_real))
            print('D_fake : %g' % np.mean(_D_fake))
            print('loss_d', _loss_d)
            print('loss_g', _loss_g)

        if i % 1000 == 0:
            _global_step = sess.run(global_step)
            saver.save(sess, checkpoint_dir+ 'model.ckpt', global_step = _global_step)

    #samples = sess.run([G], feed_dict={z:noise})
    #save_generated_samples(samples)
    real, condition = get_next_batch(batch_size)
    noise_condition = generate_ordered_conditions()
    noise = generate_random_normal_vector()

    save_image = tf.summary.image('generated', tf.multiply(tf.add(tf.reshape(G,[-1, 28, 28, 1]),1),127.5), max_outputs=30)
    image_summary = sess.run(save_image, feed_dict={z:noise, y:noise_condition})
    writer.add_summary(image_summary)
    print('write generated samples')

    save_gt = tf.summary.image('ground_truth', tf.multiply(tf.add(tf.reshape(x, [-1, 28, 28, 1]),1),127.5), max_outputs=30)
    image_summary = sess.run(save_gt, feed_dict={x:real, y:condition})
    writer.add_summary(image_summary)
    print('write ground truth')

def save_generated_samples(samples):
    s = np.array(samples[0])
    s += 1.0
    s *= 127.5
    s = s.astype('uint8')
    plt.imshow(s)
    plt.savefig('generated.png')



if __name__ == '__main__':
    main()









