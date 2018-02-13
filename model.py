from __future__ import division
import os
import sys
import time
import reader
import random
from ops import *
import scipy.misc
import numpy as np
import tensorflow as tf
from six.moves import xrange
from skimage.measure import compare_psnr
from skimage.measure import compare_mse

F = tf.app.flags.FLAGS

class layout_GAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.ngf = 128
        self.ndf = 64
        self.nt = 128
        self.k_dim = 16
        self.image_shape = [F.output_size, F.output_size, 3]
        self.build_model()
        self.iterations = 11788 // F.batch_size
        if F.output_size == 64:
            self.is_crop = True
        else:
            self.is_crop = False

    def build_model(self):
        # main method for training the conditional GAN
        if F.use_tfrecords == True:
            # load images from tfrecords + queue thread runner for better GPU utilization
            tfrecords_filename = ['train_records/' + x for x in os.listdir('train_records/')]
            filename_queue = tf.train.string_input_producer(
                                tfrecords_filename, num_epochs=100)


            self.images, self.keypoints, self.text_emb = reader.read_and_decode(filename_queue, F.batch_size)

            if F.output_size == 64:
                self.images = tf.image.resize_images(self.images, (64, 64))
                self.keypoints = tf.image.resize_images(self.keypoints, (64, 64))

            self.images = (self.images / 127.5) - 1
            # self.keypoints = (self.keypoints / 255.0)

        else:    
            self.images = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images')
            self.keypoints = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, F.c_dim], name='keypts')
            self.text_emb = tf.placeholder(tf.float32, [F.batch_size, 1024], name='text_emb')

        self.is_training = tf.placeholder(tf.bool, name='is_training')        
        self.z_gen = tf.placeholder(tf.float32, [F.batch_size, F.z_dim], name='z')

        self.G = self.generator(self.z_gen, self.keypoints, self.text_emb)
        self.D, self.D_logits = self.discriminator(self.images, self.keypoints, self.text_emb, reuse=False)
        self.D_, self.D_logits_, = self.discriminator(self.G, self.keypoints, self.text_emb, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_actual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # create summaries  for Tensorboard visualization
        tf.summary.scalar('disc_loss', self.d_loss)
        tf.summary.scalar('disc_loss_real', self.d_loss_real)
        tf.summary.scalar('disc_loss_fake', self.d_loss_fake)
        tf.summary.scalar('gen_loss', self.g_loss_actual)

        self.g_loss = tf.constant(0) 

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'D/d_' in var.name]
        self.g_vars = [var for var in t_vars if 'G/g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        # main method for training conditonal GAN

        # data = dataset()
        global_step = tf.placeholder(tf.int32, [], name="global_step_iterations")

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        
        self.summary_op = tf.summary.merge_all()

        d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G)\
          .minimize(self.g_loss_actual, var_list=self.g_vars)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        start_time = time.time()

        if F.load_chkpt:
            try:
                self.load(F.checkpoint_dir)
                print(" [*] Checkpoint Load Success !!!")
            except:
                print(" [!] Checkpoint Load failed !!!!")
        else:
            print(" [*] Not Loaded")

        self.ra, self.rb = -1, 1
        counter = 1
        step = 1
        idx = 1

        writer = tf.summary.FileWriter(F.log_dir, graph=tf.get_default_graph())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            while not coord.should_stop():
                start_time = time.time()
                step += 1

                # batch_iter = data.batch()

                for iteration in range(self.iterations):
                    # sample a noise vector 
                    sample_z_gen = np.random.uniform(
                            self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)

                    # Update D network
                    iters = 1
                    feed_dict = {
                      # self.images: zip(*sample_images)[0], 
                      # self.text_emb: zip(*sample_images)[1],
                      # self.real_labels: zip(*sample_images)[2],
                      self.z_gen: sample_z_gen,
                      self.is_training: True
                    }

                    train_summary, _, dloss, errD_fake, errD_real = self.sess.run(
                            [self.summary_op, d_optim,  self.d_loss, self.d_loss_fake, self.d_loss_real],
                            feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: True})
                    writer.add_summary(train_summary, counter)

                    # Update G network
                    iters = 1  # can play around 
                    for iter_gen in range(iters):
                        sample_z_gen = np.random.uniform(self.ra, self.rb,
                            [F.batch_size, F.z_dim]).astype(np.float32)
                        _,  gloss, dloss = self.sess.run(
                            [g_optim,  self.g_loss_actual, self.d_loss],
                            feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: True})
                       
                    lrateD = learning_rate_D.eval({global_step: counter})
                    lrateG = learning_rate_G.eval({global_step: counter})

                    print(("Iteration: [%6d] lrateD:%.2e lrateG:%.2e d_loss_f:%.8f d_loss_r:%.8f " +
                          "g_loss_act:%.8f")
                          % (idx, lrateD, lrateG, errD_fake, errD_real, gloss))

                    # peridically save generated images with corresponding checkpoints

                    if np.mod(counter, F.sampleInterval) == 0:
                        sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
                        samples, key_pts,  d_loss, g_loss_actual = self.sess.run(
                            [self.G, self.keypoints,  self.d_loss, self.g_loss_actual],
                            feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: False}
                        )
                        save_images(samples, [8, 8],
                                    F.sample_dir + "/sample_" + str(counter) + ".png")
                        save_images(key_pts,  [8, 8],  
                                    F.sample_dir + "/samplemsk_" + str(counter) + ".png")
                        print("new samples stored!!")
                     
                    # periodically save checkpoints for future loading
                    if np.mod(counter, F.saveInterval) == 0:
                        self.save(F.checkpoint_dir, counter)
                        print("Checkpoint saved successfully !!!")

                    counter += 1
                    idx += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (F.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

    def same_z_diff_keypoints(self):
        # call this while trying to show generated samples for same z-vector but different 
        # keypoint maps
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(F.checkpoint_dir)
        assert(isLoaded)

        files = os.listdir('test_images/')
        imgs = [x for x in files if 'im' in x]
        keys = [x.replace('img', 'ky') for x in imgs][:64]

        for i in range(200):
            shuffle(files)
            imgs = [x for x in files if 'im' in x]
            keys = [x.replace('img', 'ky') for x in imgs][:64]
            z_new = np.random.uniform(-1, 1, size=(F.z_dim))
            batch_keypoints = np.array([get_image('test_images/' + batch_file, F.output_size, is_crop=self.is_crop)
                         for batch_file in keys]).astype(np.float32)

            fd = {
                self.z_gen: [z_new] * F.batch_size,
                self.keypoints: batch_keypoints,
                self.is_training: False
            }
            G_imgs = self.sess.run(self.G, feed_dict=fd)

            save_images(G_imgs, [8, 8], 'experiments/same_z_diff_k_image_' + str(i) + '.png')
            save_images(batch_keypoints, [8, 8], 'experiments/same_z_diff_k_keypts_' + str(i) + '.png')

    def diff_z_same_keypoints(self):
        # call this while trying to show generated samples for different z-vectors but
        # same keypoint map

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(F.checkpoint_dir)
        assert(isLoaded)

        files = os.listdir('test_images/')
        imgs = [x for x in files if 'im' in x]
        keys = [x.replace('img', 'ky') for x in imgs][:64]
  
        batch_keypoints = np.array([get_image('test_images/' + batch_file, F.output_size, is_crop=self.is_crop)
                     for batch_file in keys]).astype(np.float32)

        for i in range(200):
            z_new = np.random.uniform(-1, 1, size=(F.batch_size, F.z_dim))
            keypoints = np.array([batch_keypoints[(i * 7) % 64]] * F.batch_size)
            fd = {
                self.z_gen: z_new,
                self.keypoints: keypoints,
                self.is_training: False
            }
            G_imgs = self.sess.run(self.G, feed_dict=fd)

            save_images(G_imgs, [8, 8], 'experiments/diff_z_same_k_image_' + str(i) + '.png')
            save_images(keypoints, [8, 8], 'experiments/diff_z_same_k_keypts_' + str(i) + '.png')  

    def discriminator(self, image, keypoints, text_emb, reuse=False):
        with tf.variable_scope('D'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            dim = 64
            image = tf.concat([image, keypoints], 3)
            if F.output_size == 128:
                  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
                  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv'), self.is_training))
                  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv'), self.is_training))
                  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv'), self.is_training))
                  h4 = lrelu(batch_norm(name='d_bn4')(conv2d(h3, dim * 16, name='d_h4_conv'), self.is_training))
                  h4 = tf.reshape(h4, [F.batch_size, -1])
                  h4 = tf.concat([h4, text_emb], 1)
                  h4 = linear(h4, 128, 'd_h4_lin')
                  h5 = linear(h4, 1, 'd_h5_lin')
                  return tf.nn.sigmoid(h5), h5

            else:
                  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
                  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv'), self.is_training))
                  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv'), self.is_training))
                  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv'), self.is_training))
                  h4 = tf.reshape(h3, [F.batch_size, -1])
                  h4 = tf.concat([h4, text_emb], 1)
                  h4 = linear(h4, 128, 'd_h4_lin')
                  h5 = linear(h4, 1, 'd_h5_lin')
                  return tf.nn.sigmoid(h5), h5

    def generator(self, z, keypoints, text_emb):
        dim = 64
        k = 5
        with tf.variable_scope("G"):
              s2, s4, s8, s16 = int(F.output_size / 2), int(F.output_size / 4), int(F.output_size / 8), int(F.output_size / 16)
              text_emb = linear(text_emb, 128, 'g_text_emb_lin')

              z = tf.concat([z, text_emb], 1)
              z = tf.reshape(z, [F.batch_size, 1, 1, 228])
              z = tf.tile(z, [1, F.output_size, F.output_size, 1])
              z = tf.concat([z, keypoints], 3)

              h0 = z
            
              h1 = tf.nn.relu(batch_norm(name='g_bn1')(conv2d(h0, dim * 2, 5, 5, 1, 1, name='g_h1_conv'), self.is_training))
              h2 = tf.nn.relu(batch_norm(name='g_bn2')(conv2d(h1, dim * 2, k, k, 2, 2, name='g_h2_conv'), self.is_training))
              h3 = tf.nn.relu(batch_norm(name='g_bn3')(conv2d(h2, dim * 4, k, k, 2, 2, name='g_h3_conv'), self.is_training))
              h4 = tf.nn.relu(batch_norm(name='g_bn4')(conv2d(h3, dim * 8, k, k, 2, 2, name='g_h4_conv'), self.is_training))
              h5 = tf.nn.relu(batch_norm(name='g_bn5')(conv2d(h4, dim * 16, k, k, 2, 2, name='g_h5_conv'), self.is_training))

              h6 = deconv2d(h5, [F.batch_size, s8, s8, dim * 8], k, k, 2, 2, name = 'g_deconv1')
              h6 = tf.nn.relu(batch_norm(name = 'g_bn6')(h6, self.is_training))
                      
              h7 = deconv2d(h6, [F.batch_size, s4, s4, dim * 4], k, k, 2, 2, name = 'g_deconv2')
              h7 = tf.nn.relu(batch_norm(name = 'g_bn7')(h7, self.is_training))

              h8 = deconv2d(h7, [F.batch_size, s2, s2, dim * 2], k, k, 2, 2, name = 'g_deconv4')
              h8 = tf.nn.relu(batch_norm(name = 'g_bn8')(h8, self.is_training))

              h9 = deconv2d(h8, [F.batch_size, F.output_size, F.output_size, 3], k, k, 2, 2, name ='g_hdeconv5')
              h9 = tf.nn.tanh(h9, name = 'g_tanh')
              return h9
              
    def save(self, checkpoint_dir, step=0):
        model_name = "model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
