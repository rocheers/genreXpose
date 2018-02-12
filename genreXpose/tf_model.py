import numpy as np
import math
import tensorflow as tf
from utils import read_mfcc_file, genre_list


def batch_data(X, Y, size=32):
    assert X.shape[0] == Y.shape[0]

    i = 0
    while True:
        if i < X.shape[0] < i + size:
            new_i = size - (X.shape[0] - i)
            yield (np.concatenate((X[i:], X[:new_i]), axis=0),
                   np.concatenate((Y[i:], Y[:new_i]), axis=0))
            i = new_i
        else:
            yield (X[i:i + size], Y[i:i + size])
        i += size


def main():
    tf.set_random_seed(1)
    train_X, train_y = read_mfcc_file(test=False, shuffle=True)
    test_X, test_y = read_mfcc_file(test=True, shuffle=True)

    print("Tensorflow version:", tf.__version__)

    train_y_onehot = np.zeros((train_y.size, train_y.max() + 1))
    train_y_onehot[np.arange(train_y.size), train_y] = 1

    test_y_onehot = np.zeros((test_y.size, test_y.max() + 1))
    test_y_onehot[np.arange(test_y.size), test_y] = 1

    X = tf.placeholder(tf.float32, [None, 20])
    y = tf.placeholder(tf.float32, [None, 4])

    lr = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_normal([20, 10]) * 0.01)
    W2 = tf.Variable(tf.random_normal([10, 4]) * 0.01)
    # W1 = tf.Variable(tf.truncated_normal([20, 10], stddev=0.1))
    # W2 = tf.Variable(tf.truncated_normal([10, 4], stddev=0.1))
    b1 = tf.Variable(tf.zeros([10]))
    b2 = tf.Variable(tf.zeros([4]))

    h1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    h2 = tf.nn.softmax(tf.matmul(h1, W2) + b2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=h2))

    # train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cost)
    train_step = tf.train.AdamOptimizer(lr).minimize(cost)

    feed_dict = {X: test_X, y: test_y_onehot}
    # feed_dict = {X: train_X, y: train_y_onehot}

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(2000):
        batch_X, batch_y = next(batch_data(train_X, train_y_onehot, size=32))

        max_learning_rate = 0.02
        min_learning_rate = 0.001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        _, loss_value = sess.run([train_step, cost], feed_dict={X: batch_X, y: batch_y, lr: learning_rate})
        if i % 100 == 0:
            print("Loss value: %.4f" % loss_value)

    correct_prediction = tf.equal(tf.argmax(h2, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict=feed_dict))


if __name__ == '__main__':
    # train_X, train_y = read_mfcc_file(test=False, shuffle=True)
    # a, b = next(batch_data(train_X, train_y))
    main()
