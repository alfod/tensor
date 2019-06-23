import tensorflow as tf

v = tf.Variable([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
re = tf.reduce_sum(v, axis=2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(re))

'''
import tensorflow as tf
x = tf.Variable(tf.constant(0.1, shape = [10]))
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(x))
    
[ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
'''
