from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import scipy.misc

import tensorflow as tf

def read_and_decode(filename_queue, batch_size=64, emb_shape=1024,
                    img_height=128, img_width=128, depth=3,
                    num_threads=16,
                    capacity=10000,
                    min_after_dequeue=1000):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string)
        'mask_raw': tf.FixedLenFeature([], tf.string)
        'emb_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    mask  = tf.decode_raw(features['mask_raw'], tf.uint8)
    emb = tf.decode_raw(features['emb_raw'], tf.float32)

    image_shape = tf.stack([img_height, img_width, depth])
    mask_shape = tf.stack([img_height, img_width, 1])
    emb_shape = tf.stack([emb_shape])

    image = tf.reshape(tf.cast(image, tf.float32), image_shape)
    mask = tf.reshape(tf.cast(mask, tf.float32), mask_shape)
    emb = tf.reshape(emb, emb_shape)

    images, mask, emb = tf.train.shuffle_batch([image, mask, emb],
                                                batch_size=batch_size,
                                                num_threads=num_threads,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue
                                               )
    
    return images, mask, emb
