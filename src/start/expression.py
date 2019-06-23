import tensorflow as tf
import numpy as np

const_2 = tf.constant(2.0, dtype=tf.float32, name="const_2")
# b = tf.Variable(2.0, "b")
c = tf.Variable(1.0, "c")
b = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="p")

add1 = tf.add(const_2, c)
add2 = tf.add(b, c)
mul = tf.multiply(add1, add2)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(mul, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print(result[1])
