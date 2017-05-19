from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.framework import get_or_create_global_step
from utils.yaleface import YaleFace



os.environ["CUDA_VISIBLE_DEVICES"] = "4"
layers = tf.contrib.layers
slim = tf.contrib.slim


batch_size = 128
noise_size = 100
learning_rate = 2e-4
dropout_rate = 1.0
height = 32
width = 32
epsilon = 0
num_epoch = 10
num_k = 5
reg_coef = 0.1
tb_dir = '/data/tensorboard_log/leewk92/gan/dcgan/yaleface/'
checkpoint_dir = './checkpoints/dcgan_face/'
is_training = True

#mnist = input_data.read_data_sets('../../mnist/', one_hot = True)
yaleface = YaleFace(height=32, width=32)


def generator(z):
    with tf.variable_scope('generator'):
        fc = layers.fully_connected(z, 1000, activation_fn=None,
                                    weights_initializer=layers.xavier_initializer(),
                                    scope='fully')
        fc = layers.batch_norm(fc, is_training=is_training)
        reshaped = tf.reshape(fc, [batch_size, 1, 1, 1000])
        with tf.variable_scope('transposed_conv2d'):
            with slim.arg_scope([layers.conv2d_transpose],
                                weights_initializer =
                                tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                activation_fn = None,
                                weights_regularizer =
                                layers.l2_regularizer(0.)
                                ):
                #z = layers.batch_norm(z, is_training=is_training)
                conv1 = layers.conv2d_transpose(reshaped, num_outputs=512,
                                                kernel_size=[5,5],
                                                stride=[2,2],
                                                scope='conv1')
                #conv1 = tf.nn.relu(layers.batch_norm(conv1, is_training=is_training))
                conv1 = tf.nn.relu(layers.batch_norm(conv1, is_training=is_training))
                conv2 = layers.conv2d_transpose(conv1, num_outputs=256,
                                                kernel_size=[5,5],
                                                stride=[2,2],
                                                scope='conv2')
                #conv2 = tf.nn.relu(layers.batch_norm(conv2, is_training=is_training))
                conv2 = tf.nn.relu(layers.batch_norm(conv2, is_training=is_training))
                conv3 = layers.conv2d_transpose(conv2, num_outputs=128,
                                                kernel_size=[5,5],
                                                stride=[2,2],
                                                scope='conv3')
                #conv3 = tf.nn.relu(layers.batch_norm(conv3, is_training=is_training))
                conv3 = tf.nn.relu(layers.batch_norm(conv3, is_training=is_training))
                conv4 = layers.conv2d_transpose(conv3, num_outputs=64,
                                                kernel_size=[5,5],
                                                stride=[2,2],
                                                scope='conv4')
                #conv4 = tf.nn.relu(layers.batch_norm(conv4, is_training=is_training))
                conv4 = tf.nn.relu(layers.batch_norm(conv4, is_training=is_training))
                generated = layers.conv2d_transpose(conv4, num_outputs=1,
                                                    kernel_size=[5,5],
                                                    stride=[2,2],
                                                    scope='generated',
                                                    activation_fn=tf.nn.tanh)

    return generated


def discriminator(x):
    x = tf.reshape(x, [batch_size, 32, 32, 1])
    with tf.variable_scope('discriminator'):
        with tf.variable_scope('conv2d'):
            with slim.arg_scope([layers.conv2d],
                                activation_fn = None,
                                weights_regularizer =
                                layers.l2_regularizer(reg_coef),
                                weights_initializer = \
                                tf.random_normal_initializer(mean=0.0, stddev=0.02)):
                #x = layers.batch_norm(x, is_training=is_training, scale=True)
                conv1 = layers.conv2d(x, 512, [3,3], 2, padding='SAME', scope='conv1')
                conv1 = lrelu(layers.batch_norm(conv1, is_training=is_training))
                conv2 = layers.conv2d(conv1, 256, [3,3], 2, padding='SAME', scope='conv2')
                conv2 = lrelu(layers.batch_norm(conv2, is_training=is_training))
                conv3 = layers.conv2d(conv2, 128, [3,3], 2, padding='SAME', scope='conv3')
                conv3 = lrelu(layers.batch_norm(conv3, is_training=is_training))
                conv4 = layers.conv2d(conv3, 64, [3,3], 2, padding='SAME', scope='conv4')
                conv4 = lrelu(layers.batch_norm(conv4, is_training=is_training))
                score = layers.conv2d(conv4, 1, [3,3], 2, padding='SAME', scope='score', activation_fn=tf.nn.sigmoid)
                #score = tf.nn.sigmoid(layers.batch_norm(score, is_training=is_training, scale=True))
                score = tf.squeeze(score, [1,2,3])
                score = tf.multiply(score, 0.99)
    return score


def get_next_batch(batch_size):
    samples = yaleface.next_batch(batch_size)
    return samples

def generate_random_normal_vector():
    noise = np.random.randn(batch_size, noise_size)
    #noise = np.random.uniform(-1, 1, [batch_size, noise_size]).astype(np.float32)
    return noise


def generate_linspaced_vector():
    start = np.random.randn(noise_size)
    end = np.random.randn(noise_size)
    noise = np.zeros((batch_size, noise_size), dtype=np.float32)
    for i in range(noise_size):
        noise[:,i] =np.linspace(start[i], end[i], num=batch_size)

    return noise

def lrelu(tensor, leak=0.2):
    return tf.maximum(tensor, leak*tensor)

def main():

    global_step = get_or_create_global_step()
    z = tf.placeholder(tf.float32, [batch_size, noise_size])
    x = tf.placeholder(tf.float32, [batch_size, height, width])
    x_reshaped = tf.reshape(x, [batch_size, height, width, 1])
    x_resized = tf.image.resize_images(x_reshaped, [32,32])

    G = generator(z)
    D_real = discriminator(x_resized)
    D_fake = discriminator(G)
    var_generator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='generator')
    var_discriminator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='discriminator')

    loss_discriminator = tf.reduce_mean(-tf.log(D_real + epsilon) - tf.log(1-D_fake + epsilon),
                                         axis=0) + get_reg_loss()
    #loss_generator = tf.reduce_mean(tf.log(1-D_fake), axis=0)
    loss_generator = tf.reduce_mean(-tf.log(D_fake), axis=0)

    optimize_discriminator = tf.train.AdamOptimizer(learning_rate*0.1, beta1=0.5).minimize(
        loss=loss_discriminator, var_list=var_discriminator, global_step=global_step)
    optimize_generator = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(
        loss=loss_generator, var_list=var_generator, global_step=global_step)
    """
    optimizer_discriminator = tf.train.AdamOptimizer(learning_rate*0.2, beta1=0.9)
    grads_and_vars_d = optimizer_discriminator.compute_gradients(
        loss=loss_discriminator, var_list=var_discriminator)
    clipped_grads_and_vars_d = [(tf.clip_by_norm(grad, 5.0),var) for grad, var
                              in grads_and_vars_d]
    optimize_discriminator = optimizer_discriminator.apply_gradients(
                                clipped_grads_and_vars_d,
                                global_step=global_step)

    optimizer_generator= tf.train.AdamOptimizer(learning_rate, beta1=0.9)
    grads_and_vars_g = optimizer_generator.compute_gradients(
        loss=loss_generator, var_list=var_generator)
    clipped_grads_and_vars_g = [(tf.clip_by_norm(grad, 5.0),var) for grad, var
                                in grads_and_vars_g]
    optimize_generator = optimizer_generator.apply_gradients(
                            clipped_grads_and_vars_g,
                            global_step=global_step)
    """

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
    for i in range(num_epoch):
        for k in range(num_k):
            real = get_next_batch(batch_size)
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
                                            feed_dict={x:real, z:noise})
        noise = generate_random_normal_vector()
        _D_fake,_loss_g, _ = sess.run([D_fake,loss_generator,
                                       optimize_generator
                                       ],
                                      feed_dict={z:noise})
        if i % 100 == 0:
            print('%g th step' % i)
            print('D_real : %g' % np.mean(_D_real))
            print('D_fake : %g' % np.mean(_D_fake))
            print('loss_d', _loss_d)
            print('loss_g', _loss_g)
            print('G', np.mean(sess.run(G, feed_dict={z:noise})))

        if i % 1000 == 1:
            _global_step = sess.run(global_step)
            saver.save(sess, checkpoint_dir+'model.ckpt', global_step = _global_step)

    #samples = sess.run([G], feed_dict={z:noise})
    #save_generated_samples(samples)
    real = get_next_batch(batch_size)
    #noise = generate_random_normal_vector()
    noise = generate_linspaced_vector()

    save_image = tf.summary.image('generated', tf.multiply(tf.add(G, 1),127.5), max_outputs=batch_size)
    image_summary = sess.run(save_image, feed_dict={z:noise})
    writer.add_summary(image_summary)
    print('write generated samples')

    save_gt = tf.summary.image('ground_truth', tf.multiply(tf.add(x_resized, 1),127.5), max_outputs=30)
    image_summary = sess.run(save_gt, feed_dict={x:real})
    writer.add_summary(image_summary)
    print('write ground truth')


def get_reg_loss():
    with tf.variable_scope('reg_loss'):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.zeros(shape=())
        for rl in reg_losses:
            reg_loss = tf.add(rl, reg_loss)
    return reg_loss



def save_generated_samples(samples):
    s = np.array(samples[0])
    s += 1.0
    s *= 127.5
    s = s.astype('uint8')
    plt.imshow(s)
    plt.savefig('generated.png')



if __name__ == '__main__':
    main()









