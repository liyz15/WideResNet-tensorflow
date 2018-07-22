import tensorflow as tf


def residual_block(x, is_training, in_nb_filters=16, nb_filters=16, stride=1, drop_prob=0.3, use_conv=False):
    shortcut = x
    strides = (stride, stride)
    if nb_filters != in_nb_filters:
        if use_conv:
            shortcut = tf.layers.conv2d(x, nb_filters, (1, 1), strides=strides, padding='valid', name='conv0')
        else:
            shortcut = tf.layers.average_pooling2d(x, pool_size=strides, strides=strides, padding='valid', name='pool0')
            shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0],
                              [(nb_filters-in_nb_filters)//2, (nb_filters-in_nb_filters)//2]])
    x = tf.layers.batch_normalization(x, axis=-1, training=is_training, name='batch1')
    x = tf.nn.relu(x, name='relu1')
    x = tf.layers.conv2d(x, nb_filters, (3, 3), strides=strides, padding='same', name='conv1')
    x = tf.layers.batch_normalization(x, axis=-1, training=is_training, name='batch2')
    x = tf.nn.relu(x, name='relu2')
    x = tf.layers.dropout(x, rate=drop_prob, training=is_training, name='dropout1')
    x = tf.layers.conv2d(x, nb_filters, (3, 3), strides=(1, 1), padding='same', name='conv2')
    x = tf.add(x, shortcut)
    return x


def wide_residual_network(x, is_training, params):
    assert 'depth' in params, 'depth must in params'
    assert 'width' in params, 'width must in params'
    assert 'drop_prob' in params, 'drop_prob must in params'
    assert 'num_classes' in params, 'num_classes must in params'

    depth = params['depth']
    width = params['width']
    drop_prob = params['drop_prob']
    # if use_conv, a 1*1 conv2d will be used for downsampling between groups
    if 'use_conv' in params:
        use_conv = params['use_conv']
    else:
        use_conv = False
    assert (depth - 4) % 6 == 0
    num_residual_units = (depth - 4) // 6
    nb_filters = [x * width for x in [16, 32, 64]]

    x = tf.layers.conv2d(x, 16, 3, strides=(1, 1), padding='same', name='conv')
    in_nb_filters = 16
    for i in range(0, num_residual_units):
        with tf.variable_scope('group_1_{}'.format(i+1)):
            x = residual_block(x,
                               is_training=is_training,
                               in_nb_filters=in_nb_filters,
                               nb_filters=nb_filters[0],
                               stride=1,
                               drop_prob=drop_prob,
                               use_conv=use_conv)
            in_nb_filters = nb_filters[0]
    for i in range(0, num_residual_units):
        if i == 0:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope('group_2_{}'.format(i+1)):
            x = residual_block(x,
                               is_training=is_training,
                               in_nb_filters=in_nb_filters,
                               nb_filters=nb_filters[1],
                               stride=stride,
                               drop_prob=drop_prob,
                               use_conv=use_conv)
            in_nb_filters = nb_filters[1]
    for i in range(0, num_residual_units):
        if i == 0:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope('group_3_{}'.format(i+1)):
            x = residual_block(x,
                               is_training=is_training,
                               in_nb_filters=in_nb_filters,
                               nb_filters=nb_filters[2],
                               stride=stride,
                               drop_prob=drop_prob,
                               use_conv=use_conv)
            in_nb_filters = nb_filters[2]
    x = tf.layers.batch_normalization(x, axis=-1, training=is_training, name='bn')
    x = tf.nn.relu(x, name='relu')
    x = tf.layers.average_pooling2d(x, pool_size=(8, 8), strides=(1, 1), padding='valid', name='pool')
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, params['num_classes'], name='fc')
    return x

