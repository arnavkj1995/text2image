from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import scipy.misc

import tensorflow as tf

def read_and_decode(filename_queue, batch_size=64, emb_shape=2048,
                    img_height=128, img_width=128, depth=3,
                    num_threads=16,
                    capacity=100,
                    min_after_dequeue=10):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string),
        'emb_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    mask  = tf.decode_raw(features['mask_raw'],tf.uint8)
    emb = tf.decode_raw(features['emb_raw'], tf.float32)

    print (emb.get_shape())
    image_shape = tf.stack([img_height, img_width, depth])
    mask_shape = tf.stack([img_height, img_width, 1])
    emb_shape = tf.stack([emb_shape])

    image = tf.reshape(tf.cast(image, tf.float32), image_shape)
    mask = tf.reshape(tf.cast(mask, tf.float32), mask_shape)
    emb.set_shape([1024])
    # emb = tf.reshape(emb, emb_shape)

    images, mask, emb = tf.train.shuffle_batch([image, mask, emb],
                                                batch_size=batch_size,
                                                num_threads=num_threads,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue
                                               )
    
    return images, mask, emb

if __name__ =='__main__':
    tfrecords_filename = ['train_records/cub_2.tfrecords', 'train_records/cub_1.tfrecords', 'train_records/cub_0.tfrecords', 'train_records/cub_4.tfrecords']
    filename_queue = tf.train.string_input_producer(
                            tfrecords_filename, num_epochs=None)

    imgs, masks, embs = read_and_decode(filename_queue, 64)

    with tf.Session() as sess:
      # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # time.sleep(5)
        while not coord.should_stop():
            # try:
            # sleep(100)
            for i in range(2048):
                # Retrieve a single instance:
                img, mask, emb = sess.run([imgs, masks, embs])
                print (img.shape, mask.shape, emb.shape)
            print ('Done')

        # except tf.errors.OutOfRangeError:
        #   print('Done training for')
        # finally:
        #   # When done, ask the threads to stop.
        #   coord.request_stop()

        coord.request_stop()
        coord.join(threads)
