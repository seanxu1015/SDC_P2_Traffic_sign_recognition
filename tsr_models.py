import tensorflow as tf


def conv(x, W, b, strides=(1, 1, 1, 1), padding='SAME'):
    y = tf.nn.conv2d(x, W, strides=strides, padding=padding)
    y = tf.nn.bias_add(y, b)
    return y


def tsr_model_01(x, dp):
    #x1 = tf.nn.dropout(x, dp)
    x1 = x
    W1 = tf.get_variable('W1', (3, 3, 3, 32))
    b1 = tf.get_variable('b1', (32,))
    conv1 = conv(x1, W1, b1)
    relu1 = tf.nn.relu(conv1)

    x2 = tf.nn.dropout(relu1, dp)
    #x2 = relu1
    W2 = tf.get_variable('W2', (3, 3, 32, 32))
    b2 = tf.get_variable('b2', (32,))
    conv2 = conv(x2, W2, b2)
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2, ksize=(1, 2, 2, 1), 
            strides=(1, 2, 2, 1), padding='SAME')
            
    x3 = tf.nn.dropout(pool2, dp)
    #x3 = pool2
    W3 = tf.get_variable('W3', (3, 3, 32, 32))
    b3 = tf.get_variable('b3', (32,))
    conv3 = conv(x3, W3, b3)
    relu3 = tf.nn.relu(conv3)

    x4 = tf.nn.dropout(relu3, dp)
    #x4 = relu3
    W4 = tf.get_variable('W4', (3, 3, 32, 32))
    b4 = tf.get_variable('b4', (32,))
    conv4 = conv(x4, W4, b4)
    relu4 = tf.nn.relu(conv4)
    pool4 = tf.nn.max_pool(relu4, ksize=(1, 2, 2, 1), 
            strides=(1, 2, 2, 1), padding='SAME')

    x5 = tf.nn.dropout(pool4, dp)
    #x5 = pool4
    W5 = tf.get_variable('W5', (3, 3, 32, 32))
    b5 = tf.get_variable('b5', (32,))
    conv5 = conv(x5, W5, b5)
    relu5 = tf.nn.relu(conv5)

    x6_1 = tf.reshape(relu3, (-1, 8192))
    x6_2 = tf.reshape(relu5, (-1, 2048))
    x6 = tf.nn.dropout(tf.concat(1, [x6_1, x6_2]), dp)
    W6 = tf.get_variable('W6', (10240, 10240))
    b6 = tf.get_variable('b6', (10240,))
    fc6 = tf.matmul(x6, W6) + b6
    relu6 = tf.nn.relu(fc6)

    x7 = tf.nn.dropout(relu6, dp)
    W7 = tf.get_variable('W9', (10240, 43))
    b7 = tf.get_variable('b9', (43,))
    fc7 = tf.matmul(x7, W7) + b7
    logits = tf.nn.softmax(fc7)

    return logits


def tsr_model_02(x, dp):
    x1 = tf.nn.dropout(x, dp)
    W1 = tf.get_variable('W1', (3, 3, 3, 32))
    b1 = tf.get_variable('b1', (32,))
    conv1 = conv(x1, W1, b1)
    relu1 = tf.nn.relu(conv1)

    x2 = tf.nn.dropout(relu1, dp)
    W2 = tf.get_variable('W2', (3, 3, 32, 32))
    b2 = tf.get_variable('b2', (32,))
    conv2 = conv(x2, W2, b2)
    relu2 = tf.nn.relu(conv2)
            
    x5 = tf.nn.dropout(relu2, dp)
    W5 = tf.get_variable('W5', (3, 3, 32, 32))
    b5 = tf.get_variable('b5', (32,))
    conv5 = conv(x5, W5, b5)
    relu5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(relu5, ksize=(1, 2, 2, 1), 
            strides=(1, 2, 2, 1), padding='SAME')
            
    x6 = tf.nn.dropout(pool5, dp)
    W6 = tf.get_variable('W6', (3, 3, 32, 32))
    b6 = tf.get_variable('b6', (32,))
    conv6 = conv(x6, W6, b6)
    relu6 = tf.nn.relu(conv6)
            
    x7 = tf.nn.dropout(relu6, dp)
    W7 = tf.get_variable('W7', (3, 3, 32, 32))
    b7 = tf.get_variable('b7', (32,))
    conv7 = conv(x7, W7, b7)
    relu7 = tf.nn.relu(conv7)

    x3 = tf.nn.dropout(relu7, dp)
    W3 = tf.get_variable('W3', (1, 1, 32, 64))
    b3 = tf.get_variable('b3', (64,))
    conv3 = conv(x3, W3, b3)
    relu3 = tf.nn.relu(conv3)
    pool3 = tf.nn.max_pool(relu3, ksize=(1, 2, 2, 1), 
            strides=(1, 2, 2, 1), padding='SAME')

    x4 = tf.reshape(pool3, (-1, 4096))
    W4 = tf.get_variable('W4', (4096, 4096))
    b4 = tf.get_variable('b4', (4096,))
    fc4 = tf.matmul(x4, W4) + b4
            
    x8 = tf.nn.dropout(fc4, dp)
    W8 = tf.get_variable('W8', (4096, 43))
    b8 = tf.get_variable('b8', (43,))
    fc8 = tf.matmul(x8, W8) + b8
    logits = tf.nn.softmax(fc8)

    return logits


def inception(x, dp, name, channels):

    with tf.variable_scope(name):

        x1 = tf.nn.dropout(x, dp)
        W1 = tf.get_variable('W1', (1, 1, channels, channels/2))
        b1 = tf.get_variable('b1', (channels/2,))
        conv1 = conv(x1, W1, b1)

        W2 = tf.get_variable('W2', (1, 1, channels, channels/2))
        b2 = tf.get_variable('b2', (channels/2,))
        conv2 = conv(x1, W2, b2)
        W3 = tf.get_variable('W3', (3, 3, channels/2, channels/4))
        b3 = tf.get_variable('b3', (channels/4,))
        conv3 = conv(conv2, W3, b3)

        W4 = tf.get_variable('W4', (1, 1, channels, channels/4))
        b4 = tf.get_variable('b4', (channels/4,))
        conv4 = conv(x1, W4, b4)
        W5 = tf.get_variable('W5', (5, 5, channels/4, channels/8))
        b5 = tf.get_variable('b5', (channels/8,))
        conv5 = conv(conv4, W5, b5)
        x6 = tf.nn.max_pool(x1, (1, 3, 3, 1), (1, 1, 1, 1), 'SAME')
        W6 = tf.get_variable('W6', (1, 1, channels, channels/8))
        b6 = tf.get_variable('b6', (channels/8,))
        conv6 = conv(x6, W6, b6)

        return tf.concat(3, [conv1, conv3, conv5, conv6])


def tsr_model_03(x, dp):

    x1 = tf.nn.dropout(x, dp)
    W1 = tf.get_variable('W1', (32, 32, 3, 32))
    b1 = tf.get_variable('b1', (32,))
    conv1 = conv(x1, W1, b1)
    relu1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(relu1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

    x2 = tf.nn.dropout(pool1, dp)
    W2 = tf.get_variable('W2', (16, 16, 32, 128))
    b2 = tf.get_variable('b2', (128,))
    conv2 = conv(x2, W2, b2)
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

    incep1 = inception(pool2, dp, 'inception01', 128)
    incep2 = inception(incep1, dp, 'inception02', 128)

    x5 = tf.reshape(incep2, (-1, 8192))
    W5 = tf.get_variable('W5', (8192, 2048))
    b5 = tf.get_variable('b5', (2048,))
    fc5 = tf.matmul(x5, W5) + b5
    relu5 = tf.nn.relu(fc5)

    incep3 = inception(incep2, dp, 'inception03', 128)
    incep4 = inception(incep3, dp, 'inception04', 128)

    x3 = tf.reshape(incep4, (-1, 8192))
    W3 = tf.get_variable('W3', (8192, 2048))
    b3 = tf.get_variable('b3', (2048,))
    fc3 = tf.matmul(x3, W3) + b3
    relu3 = tf.nn.relu(fc3)

    x4 = tf.concat(1, [tf.nn.dropout(relu3, dp), tf.nn.dropout(relu5, dp)])
    W4 = tf.get_variable('W4', (4096, 43))
    b4 = tf.get_variable('b4', (43,))
    fc4 = tf.matmul(x4, W4) + b4
    logits = tf.nn.softmax(fc4)

    return logits


def tsr_model_04(x, dp):
    
    W1 = tf.get_variable('W1', (32, 32, 3, 32))
    b1 = tf.get_variable('b1', (32,))
    conv1 = conv(x, W1, b1)
    relu1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(relu1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

    W2 = tf.get_variable('W2', (16, 16, 32, 128))
    b2 = tf.get_variable('b2', (128,))
    conv2 = conv(pool1, W2, b2)
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

    incep1 = inception(pool2, dp, 'inception01', 128)
    incep2 = inception(incep1, dp, 'inception02', 128)
    incep3 = inception(incep2, dp, 'inception03', 128)
    incep4 = inception(incep3, dp, 'inception04', 128)

    x3 = tf.reshape(incep4, (-1, 8192))
    W3 = tf.get_variable('W3', (8192, 2048))
    b3 = tf.get_variable('b3', (2048,))
    fc3 = tf.matmul(x3, W3) + b3
    relu3 = tf.nn.relu(fc3)

    x4 = tf.nn.dropout(relu3, dp)
    W4 = tf.get_variable('W4', (2048, 43))
    b4 = tf.get_variable('b4', (43,))
    fc4 = tf.matmul(x4, W4) + b4
    logits = tf.nn.softmax(fc4)

    return logits

