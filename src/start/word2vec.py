import tensorflow as tf

vocabulary_size = 10
embedding_size = 10

embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0,1.0))
