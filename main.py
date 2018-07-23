import argparse
import os

import tensorflow as tf
from wide_residual_network import wide_residual_network

tf.logging.set_verbosity(tf.logging.INFO)

def data_preprocess(is_training=True):
    def preprocess(feature):
        meanstd = {'mean': [129.3, 124.1, 112.4], 'std': [68.2, 65.4, 70.4]}
        feature = tf.cast(feature, tf.float32)
        feature = tf.divide(tf.subtract(feature, meanstd['mean']), meanstd['std'])
        if is_training:
            feature = tf.image.resize_image_with_crop_or_pad(feature, 40, 40)
            feature = tf.random_crop(feature, [32, 32, 3])
            feature = tf.image.random_flip_left_right(feature)
        return feature

    return preprocess



def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    def get_lr():
        assert len(params['lr_boundaries']) + 1 == len(params['lr_values']), 'The learning_rate setting is incorrect'
        return tf.train.piecewise_constant(tf.train.get_global_step(), params['lr_boundaries'], params['lr_values'])

    with tf.variable_scope('net'):
        logits = wide_residual_network(features, is_training, params)
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='loss'))
        with tf.variable_scope('weight_decay'):
            regular_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss = loss + params['weight_decay'] * regular_loss
    with tf.variable_scope('predictions'):
        predictions = tf.argmax(tf.nn.softmax(logits=logits), axis=-1, name='predictions')
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=get_lr(), momentum=params['momentum'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    with tf.variable_scope('accuracy'):
        accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=-1), predictions)

    logging_hook = tf.train.LoggingTensorHook({'loss' : loss, 'accuracy': accuracy[1], 'lr': get_lr()}, every_n_iter=100)

    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops={'accuracy': accuracy},
            training_hooks = [logging_hook])

params = {'depth': 28,
          'width': 10,
          'drop_prob': 0.3,
          'num_classes': 100,

          'batch_size': 128,
          'lr_boundaries': [28125, 56250, 75000],
          'lr_values': [0.1, 2e-2, 4e-3, 8e-4],
          'momentum': 0.9,
          'weight_decay': 5e-4}
(x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.cifar100.load_data(label_mode='fine')

train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_train,
        y=y_train,
        batch_size=params['batch_size'],
        num_epochs=None,
        shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_test,
        y=y_test,
        batch_size=params['batch_size'],
        num_epochs=1,
        shuffle=False)

def input_fn(is_training):
    def fn():
        if is_training:
            features, target = train_input_fn()
        else:
            features, target = test_input_fn()
        features = tf.cast(features, tf.float32)
        features = tf.map_fn(data_preprocess(is_training), features)
        target = tf.one_hot(target, depth=params['num_classes'])
        return features, target
    return fn


model = tf.estimator.Estimator(model_fn=model_fn,
                               model_dir='models/',
                               config=None,
                               params=params)

model.evaluate(input_fn=input_fn(is_training=False))
for _ in range(18):
    model.train(input_fn=input_fn(is_training=True), steps=5000)
    model.evaluate(input_fn=input_fn(is_training=False))

