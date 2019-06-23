from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
learn_rate = 0.5
epochs = 10
batch_size = 100
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
w1 = tf.Variable(tf.random_normal([784, 300], stddev=0.3), name='w1')
b1 = tf.Variable(tf.random_normal([300]), name="b1")
w2 = tf.Variable(tf.random_normal([300, 10]), name="w2")
b2 = tf.Variable(tf.random_normal([10]), name="b2")

hidden_1_in = tf.add(tf.matmul(x, w1), b1)
hidden_1_out = tf.nn.relu(hidden_1_in)
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_1_out, w2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    print("total_batch: ", total_batch)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), " cost=", "{:.3f}".format(avg_cost))
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("acc: ", acc)
    # writer.add_summary(tf.cast(acc, tf.string), epoch)
    # writer = tf.summary.FileWriter("./logs", sess.graph)
    # writer.close()
