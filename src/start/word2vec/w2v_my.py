import tensorflow as tf

import collections as cts
import random as random
import numpy as np
import math
import os
from tensorflow.contrib.tensorboard.plugins import projector

RESOURCE_FILE_PATH = "./resource/text8/text8"
VOCABUlARY_SIZE = 500
data_index = 0

__ss__: str = "s"
with open("./resource/text8/text8") as f:
    vocabulary = tf.compat.as_str(f.read()).split()

count = [['UNK', -1]]
count.extend(cts.Counter(vocabulary).most_common(VOCABUlARY_SIZE - 1))
dictionary = {}
for word, _ in count:
    dictionary[word] = len(dictionary)
print(dictionary)
data = []
unk_count = 0
for word in vocabulary:
    index = dictionary.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0][1] = unk_count
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
del vocabulary


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = cts.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        word_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(word_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):

            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


BATCH_SIZE = 128
EMBEDDING_SIZE = 128
SKIP_WINDOW = 1
NUM_SKIPS = 2
NUM_SAMPLED = 64

VALID_SIZE = 16
VALID_WINDOW = 100
VALID_EXAMPLES = np.random.choice(VALID_WINDOW, VALID_SIZE, replace=False)

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope("inputs"):
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(VALID_EXAMPLES, dtype=tf.int32)

    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(tf.random_uniform([VOCABUlARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        with tf.name_scope("weights"):
            nce_weights = tf.Variable(tf.truncated_normal([VOCABUlARY_SIZE, EMBEDDING_SIZE],
                                                          stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))

        with tf.name_scope("biases"):
            nce_biases = tf.Variable(tf.zeros([VOCABUlARY_SIZE]))

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels,
                                             inputs=embed, num_sampled=NUM_SAMPLED, num_classes=VOCABUlARY_SIZE))
    tf.summary.scalar("loss", loss)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embedding = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embedding, transpose_b=True)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

num_steps = 40000

log_dir = "./log_my"
with tf.Session(graph=graph) as session:
    writer = tf.summary.FileWriter(log_dir, session.graph)
    init.run()

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW)
        feed_dic = {train_inputs: batch_inputs, train_labels: batch_labels}
        run_meta = tf.RunMetadata()

        _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dic, run_metadata=run_meta)
        average_loss += loss_val
        writer.add_summary(summary, step)

        if step == (num_steps - 1):
            writer.add_run_metadata(run_meta, 'step%d' % step)

        if step % 20000 == 0:
            if step > 0:
                average_loss /= 2000
            print("average loss at step: %d is %f" % (step, average_loss))
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(VALID_SIZE):
                valid_word = reversed_dictionary[VALID_EXAMPLES[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s is： ' % valid_word
                for k in range(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s  %s' % (log_str, close_word)
                print(log_str)

    final_embedding = normalized_embedding.eval()

    with open(log_dir+"/metadata.tsv", 'w') as f:
        for i in range(VOCABUlARY_SIZE):
            f.write(reversed_dictionary[i] + "\n")

    saver.save(session, log_dir + "/model.ckpt")

    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = "metadata.tsv"
    projector.visualize_embeddings(writer, config)

writer.close()
