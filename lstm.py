import tensorflow as tf
vocabSize = tf.constant(12, dtype=tf.int8)
input = tf.placeholder(dtype=tf.float32, shape=[10, vocabSize])

"""Weights"""
Wf = tf.Variable(tf.zeros([]))
