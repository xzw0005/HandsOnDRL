import tensorflow as tf
import tensorflow.contrib.layers as layers

def model(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope('convnet'):
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope('action_value'):
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def dueling_model(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope('convnet'):
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope('state_value'):
            v_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            state_val = layers.fully_connected(v_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope('adv_value'):
            adv_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            adv = layers.fully_connected(adv_hidden, num_outputs=num_actions, activation_fn=None)
            adv_mean = tf.reduce_mean(adv, axis=1, keepdims=True)
            adv -= adv_mean
        return state_val + adv

def bootstrap_model(img_in, num_actions, scope, reuse=False, K=10):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope('convnet'):
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        out_list = []
        with tf.variable_scope('heads'):
            for i in range(K):
                head_scope = 'qvals_head_i'%i
                with tf.variable_scope(head_scope):
                    otemp = out
                    otemp = layers.fully_connected(otemp, num_outputs=512, activation_fn=tf.nn.relu)
                    otemp = layers.fully_connected(otemp, num_outputs=num_actions, activation_fn=None)
                out_list.append(otemp)
        return out_list
