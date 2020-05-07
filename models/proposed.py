import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader, GenDataLoader
from utils import ops
import h5py
from .base_net_metric import base_model_metric
import time
from utils.data_reader import get_pairwiselabel


class proposed_model(base_model_metric):

    def __init__(self, sess, conf):
        super(proposed_model, self).__init__(sess, conf)
        self.configure_networks()

    def GA(self, x, reuse):
        with tf.variable_scope("gen_a2b"):
            return self.inference(x, reuse=reuse)
    def GB(self, y, reuse):
        with tf.variable_scope("gen_b2a"):
          return self.inference(y, reuse=reuse)
    def D1(self, x, y, reuse):
        with tf.variable_scope('dis1'):
            d, feature = self.discriminator(x, y, reuse=reuse)
        return d, feature
    def D2(self, x, y, reuse):
        with tf.variable_scope('dis2'):
            d, feature = self.discriminator(x, y, reuse=reuse)
        return d, feature


    def D_layer(self, x, reuse):
        with tf.variable_scope('d_h') as scope:
            if reuse:
                scope.reuse_variables()
            h = tf.contrib.layers.fully_connected(x, 20, activation_fn=None, scope = '/fully') 
        return h

    def configure_networks(self):
        self.build_network()
        trainable_vars = tf.trainable_variables()
        self.g_vars = [var for var in trainable_vars if var.name.startswith('g')]
        self.d_vars = [var for var in trainable_vars if var.name.startswith('d')]
        self.g_train = tf.train.AdamOptimizer(1e-3, beta1 = 0.5, beta2 =0.999).minimize(self.g_loss_total, var_list = self.g_vars)
        self.d_train = tf.train.AdamOptimizer(2e-4, beta1 = 0.5, beta2 =0.999).minimize(self.d_loss_total, var_list = self.d_vars)  
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer()) 
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1)


    def build_network(self):
        self.cat = tf.placeholder(tf.int32, [self.conf.batch,1], name = 'cat')     #class labels
        self.pl_total = tf.placeholder(tf.float32, [None, 1], name = 'pl_total')   #pairwise label  
        self.a_p = tf.placeholder(tf.float32, self.input_shape, name='a_p')
        self.b_p = tf.placeholder(tf.float32, self.output_shape, name='b_p')
        self.a_u = tf.placeholder(tf.float32, self.input_shape, name='a_u')
        self.b_u = tf.placeholder(tf.float32, self.output_shape, name='b_u') 

        self.fake_ap = self.GB(self.b_p, False) # get generator outputs
        self.fake_bp = self.GA(self.a_p, False)
        self.fake_a = self.GB(self.b_u, True) 
        self.rec_b = self.GA(self.fake_a, True)
        self.fake_b = self.GA(self.a_u, True)
        self.rec_a = self.GB(self.fake_b, True)

        self.real_AB, self.feat_AB = self.D1(self.a_p, self.b_p, False)      #get discriminator outputs
        self.fake_AB1, self.feat_AB1 = self.D1(self.a_u, self.fake_b, True)  #reuse: share variables
        self.fake_AB2, self.feat_AB2 = self.D1(self.fake_a, self.b_u, True)
        self.D2_real, self.feat2_AB1 = self.D2(self.a_u, self.fake_b, False) 
        self.D2_fake, self.feat2_AB2 = self.D2(self.fake_a, self.b_u, True) 
        _, self.feat2_AB = self.D2(self.a_p, self.b_p, True)    #just to get the feature representation
        self.cal_loss()
    

    def cal_loss(self):
        print('concatenate [D1,D2], and calculate metric loss')
        '''first concatenate the three types of (mri, pet) pairs and pairwise labels, then calc metric loss'''
        self.featAB = tf.concat([self.feat_AB, self.feat2_AB], -1)
        self.featAB1 = tf.concat([self.feat_AB1, self.feat2_AB1], -1)
        self.featAB2 = tf.concat([self.feat_AB2, self.feat2_AB2], -1)
        ### add a fc layer
        self.featAB = self.D_layer(self.featAB, False)
        self.featAB1 = self.D_layer(self.featAB1, True)
        self.featAB2 = self.D_layer(self.featAB2, True)
        self.hidden = tf.concat((self.featAB, self.featAB1, self.featAB2), 0)
        print('hidden size', self.hidden.shape)
        self.m_loss = self.metric_loss(self.hidden, self.pl_total)
        ## ==================== generator loss ============================
        MSE_loss = tf.losses.mean_squared_error
        ## loss 1: paired + unpaired
        self.g1_loss = ops.GANLoss(self.fake_AB1, True) + ops.GANLoss(self.fake_AB2, True) 
        ## loss 2: unpaired
        self.g_cyc = ops.criterionCycle(self.rec_a, self.a_u) + ops.criterionCycle(self.rec_b, self.b_u)
        self.g2_loss = ops.GANLoss(self.D2_real, False) + ops.GANLoss(self.D2_fake, True)
        ## loss 3: paired
        self.completion = MSE_loss(self.a_p, self.fake_ap) + MSE_loss(self.b_p, self.fake_bp)
        self.g_loss_total = self.m_loss + self.g1_loss + self.g2_loss + 0.01*self.completion + 0.01*self.g_cyc
        ## ==================== discriminator loss ========================
        ## loss 1: paired + unpaired
        self.d1_loss = ops.GANLoss(self.real_AB, True) + ops.GANLoss(self.fake_AB1, False) + ops.GANLoss(self.fake_AB2, False)
        ## loss 2: unpaired
        self.d2_loss = ops.GANLoss(self.D2_real, True) + ops.GANLoss(self.D2_fake, False)
        self.d_loss_total = self.d1_loss + self.d2_loss + self.m_loss


    def metric_loss(self, hidden, pl):
        # for metric learning
        mid = int(3*self.conf.batch/2)
        h_x = hidden[:mid]
        h_y = hidden[mid:]
        dxy = tf.reduce_mean((h_x-h_y)*(h_x-h_y), 1, keepdims=True)
        cost = tf.reduce_sum( tf.maximum((1. - pl*(1.-dxy)), 0) )
        self.tmp = tf.maximum((1. - pl*(1.-dxy)),0)
        return cost


    def train(self): 
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        train_pair_reader   = H5DataLoader(self.conf.data_dir+self.conf.train_pair, model_type='paired_metric', portion=self.conf.portion)
        train_unpair_reader = H5DataLoader(self.conf.data_dir+self.conf.train_unpair, conf=self.conf, model_type='unpaired_metric')
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data, model_type='paired_metric', is_train=False)

        train_p2 = GenDataLoader(model_type='unpaired', conf=self.conf, portion=self.conf.portion)
        valid_p2 = H5DataLoader(self.conf.data_dir+self.conf.valid_data, model_type='paired', is_train=False)

        iteration = train_pair_reader.iter + self.conf.reload_step
        pre_iter = iteration
        epoch_num = self.conf.reload_step
        start_time = time.time()
        bestsim = 0
        best_acc = 0
        epochs_no_performance_gain = 0
        while epoch_num < self.conf.max_step:
            Ap, Bp, p_label   = train_pair_reader.next_batch(self.conf.batch)
            Au, Bu, up_label  = train_unpair_reader.next_batch(self.conf.batch)
            pl_lab = get_pairwiselabel( np.concatenate((p_label, up_label),0) )
            feed_dict = {self.a_p:Ap, self.b_p:Bp, self.cat:p_label, self.a_u:Au, self.b_u:Bu, self.pl_total:pl_lab}
            d_loss, m_loss_d, _ = self.sess.run([self.d_loss_total, self.m_loss, self.d_train], feed_dict=feed_dict)
            g_loss, g1, g2, m_loss_g, _, tmp = self.sess.run([self.g_loss_total, self.g1_loss, self.g2_loss, self.m_loss, self.g_train, self.tmp], feed_dict=feed_dict)

            if epoch_num % 200 == 0:
                print ('epoch %d, duration %0.2f, d_loss %0.3f, m_loss_d %0.3f, g_loss %0.3f, g1 %0.3f, g2 %0.3f, m_loss_g %0.3f' %(epoch_num, time.time()-start_time, d_loss, m_loss_d, g_loss, g1, g2, m_loss_g) )
                start_time = time.time()
            if epoch_num % 1000 == 0:
                ######################### validation costs ################################
                Ap, Bp, p_label = valid_reader.next_batch(self.conf.batch)   #### mri, pet, categ_labels
                pl_lab = get_pairwiselabel( np.concatenate((p_label, p_label, p_label),0) )
                feed_dict = {self.a_p: Ap, self.b_p: Bp, self.cat: p_label, self.a_u:Ap, self.b_u:Bp, ## tensors except for a_p, b_p are used for placeholder only
                             self.pl_total:pl_lab}
                d_loss = self.sess.run(self.d_loss_total, feed_dict=feed_dict)
                g_loss, gen_a, gen_b = self.sess.run([self.g_loss_total, self.fake_ap, self.fake_bp], feed_dict=feed_dict)
                ps = 0.
                ss = 0.
                a_p = np.squeeze(Ap, axis=-1)
                b_p = np.squeeze(Bp, axis=-1)
                gen_a = np.squeeze(gen_a, axis=-1)
                gen_b = np.squeeze(gen_b, axis=-1)
                for i, (ap, ga, bp, gb) in enumerate(zip(a_p, gen_a, b_p, gen_b)):
                    ap, bp, ga, gb = ops.normalize(ap), ops.normalize(bp), ops.normalize(ga), ops.normalize(gb)
                    ps += (ops.psnr(ap, ga) + ops.psnr(bp, gb))/2
                    ss += (ops.ssim(ap, ga) + ops.ssim(bp, gb))/2
                acc, f1, RI, purity  = self.calculate_cost(self.sess, train_p2, valid_p2)
                print ('proposed_model valid d loss %0.3f, g loss %0.3f, PSNR %0.3f, SSIM %0.3f, acc %0.4f, f1 %0.4f, RI %0.4f, purity %0.4f' %(d_loss, g_loss, ps/self.conf.batch, ss/self.conf.batch, acc, f1, RI,purity) )

                if best_acc < acc:
                    best_acc = acc
                    best_f1, best_purity, best_RI = f1, purity, RI
                    self.save(epoch_num)
                    epochs_no_performance_gain = 0
                else:
                    epochs_no_performance_gain += 1
                    if epochs_no_performance_gain >5:
                        print('stop since no improvement for %d epochs' % epochs_no_performance_gain)
                        # print('acc, f1, purity, RI &%.4f &%.4f & &%.4f &%.4f'% (best_acc, best_f1, best_purity, best_RI))
                        print('acc, f1, purity, RI [%.4f, %.4f, %.4f, %.4f]'% (best_acc, best_f1, best_purity, best_RI))
                        break

            epoch_num += 1

    

