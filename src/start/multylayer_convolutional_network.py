from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# region funcs
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


def deepnn(x):
    # Tensorboard中的命名空间，with下的都属于该空间范围
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        variable_summaries(W_conv1)
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope("poo1"):
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2)
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope("poo2"):
        h_pool2 = max_pool_2x2(h_conv2)
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return keep_prob, y_conv


# endregion
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='input_y_')

keep_prob, y_conv = deepnn(x)

with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    tf.summary.scalar('loss', cross_entropy)
with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 合并所有的summary，之后一起存入磁盘
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 将Tensorboard图像的文件写到磁盘（可以设置绝对路径）
        # 在FileWriter的构造函数中加入了sess.graph后，就不需要再使用add_graph函数传递graph了
        # 给两个graph，所以打开tensorboard页面后，相关图会有两条线
        # 观测可知模型是否适当拟合
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/test', sess.graph)
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            # 训练模型
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            # 记录训练集计算的参数
            if i % 500 == 0:
                print(i)
                train_summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                # 将summary放进协议缓冲区
                train_writer.add_summary(train_summary, i)
                # 记录测试集计算的参数
                test_batch_xs, test_batch_ys = mnist.test.next_batch(50)
                test_summary = sess.run(merged, feed_dict={x: test_batch_xs, y_: test_batch_ys, keep_prob: 1.0})
                test_writer.add_summary(test_summary, i)

        print('最终测试 准确度 %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
