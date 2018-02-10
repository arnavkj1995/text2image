from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from scipy.misc import imsave, imresize

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images, masks, embeddings, name):
    """Converts a dataset to tfrecords."""
    if images.shape[0] != masks.shape[0]:
    	raise ValueError('Images size %d does not match masks shape %d.' %
                     (images.shape[0], num_examples))

    filename = os.path.join(save_path, name + '.tfrecords')
    
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(images.shape[0]):
        image_raw = images[index].tostring()
        mask_raw = masks[index].tostring()
        emb_raw = embeddings[index].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'mask_raw': _bytes_feature(mask_raw),
            'image_raw': _bytes_feature(image_raw),
            'emb_raw': _bytes_feature(emb_raw)}))
        
        writer.write(example.SerializeToString())

    writer.close()

if __name__ =='__main__':
    save_path = 'train_records/'

    images_dir_path = 'data/train/images/'

    counter = 0

    image_list, emb_list, mask_list = [], [], []
    tfrecord_ind = 0

    for emb_ind in range(10):

        for imgs in face_image_list:
            counter += 1

            

                image_list.append(face_part)
                landmark_list.append(key_point_matrix)

        convert_to(np.asarray(image_list), np.asarray(mask_list), np.asarray(emb_list), 'cub_' + str(emb_ind))
        image_list, emb_list, mask_list = [], [], []
