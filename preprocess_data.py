from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import torchfile
import numpy as np
from skimage import io
from skimage.transform import resize
import tensorflow as tf
from scipy.misc import imsave, imresize

FLAGS = None
save_path = ''
images_dir_path = ''
images_root_path = ''
segment_root_path = ''
bbox_root = ''
out_shape = 128

image_name_bbox = {}

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

def convert_to(images, masks, embeddings, name):
	"""Converts a dataset to tfrecords."""
	print ('shapes are:: ', images.shape, masks.shape, embeddings.shape)

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

def get_total_images(base_path):
	total_imgs = 0
	files = os.listdir(base_path)
	for file in files:
		total_imgs += len(os.listdir(os.path.join(base_path, file)))
	return total_imgs

def custom_crop(img, bbox):
	imsiz = img.shape
	center_x = int((2 * bbox[0] + bbox[2]) / 2)
	center_y = int((2 * bbox[1] + bbox[3]) / 2)
	R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
	y1 = np.maximum(0, center_y - R)
	y2 = np.minimum(imsiz[0], center_y + R)
	x1 = np.maximum(0, center_x - R)
	x2 = np.minimum(imsiz[1], center_x + R)
	img_cropped = img[y1:y2, x1:x2, :]
	return img_cropped

def get_image_and_mask(image_file):
	"""Generate image and it's mask"""
	imgs_reshaped, mask_reshaped = None, None

	flag = True
	image_path = os.path.join(images_root_path, image_file)
	mask_path = os.path.join(segment_root_path, image_file.replace('jpg', 'png'))

	try:
		imgs = io.imread(image_path)
	except:
		flag = False
	try:
		mask = io.imread(mask_path)
	except:
		flag = False

	mask = np.expand_dims(mask, -1)
	bbox = image_name_bbox[image_file.split('/')[1].split('.')[0]]
	bbox = np.array(bbox).astype(np.float32)

	# Crop the image and mask
	try:
		imgs_cropped = custom_crop(imgs, bbox)
		mask_cropped = custom_crop(mask, bbox)
		imgs_reshaped = imresize(imgs_cropped, (out_shape, out_shape))
		mask_reshaped = resize(mask_cropped, (out_shape, out_shape)) * 255
	except:
		flag = False

	return flag, imgs_reshaped, mask_reshaped

if __name__ =='__main__':

	lines = open('CUB/CUB_200_2011/images.txt', 'r').readlines()
	lines = [line.split(' ')[-1] for line in lines]
	bbox_lines = open('CUB/CUB_200_2011/bounding_boxes.txt').readlines()
	bbox_lines = [line.strip().split(' ', 1)[-1] for line in bbox_lines]
	for idx, (img_path, bbox_dims) in enumerate(zip(lines, bbox_lines)):
		image_name_bbox[img_path.split('/')[1].split('.')[0]] = bbox_dims.split(' ')

	classes_dir = os.listdir('cub_icml')
	total_imgs = get_total_images('cub_icml')
	for idx in range(10):
		print ('Generating tfrecords for embedding #{}'.format(idx))
		single_tfrecord_imgs = []
		single_tfrecord_mask = []
		single_tfrecord_embs = []
		counter = 0
		for class_idx, class_dir in enumerate(classes_dir):
			files = os.listdir(os.path.join('cub_icml', class_dir))
			per_class_count = 0
			for image_idx, file in enumerate(files):
				torch_file = torchfile.load(os.path.join('cub_icml', class_dir, file))
				img_file = torch_file['img']
				embs = torch_file['txt'][idx]
				flag, img, mask = get_image_and_mask(str(img_file))

				if flag:
					# Stack the images
					single_tfrecord_imgs.append(img) #[counter] = img
					single_tfrecord_mask.append(mask) #[counter] = mask
					single_tfrecord_embs.append(embs) #[counter] = embs
					counter += 1
					per_class_count += 1

			sys.stdout.write ('{:2d}/{:2d} images processed\n'.format(per_class_count, len(files)))
		convert_to(np.asarray(single_tfrecord_imgs, dtype=np.uint8), np.asarray(single_tfrecord_mask, dtype=np.uint8), np.asarray(single_tfrecord_embs, dtype=np.float32), 'cub_' + str(idx))
