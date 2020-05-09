import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader, GenDataLoader
from utils import ops
import h5py
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import euclidean_distances
from collections import OrderedDict, Counter
import time
from utils.clustering_evaluation import RandIndex, Purity, convert_list_2_dict, rand_index_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, adjusted_mutual_info_score



class base_model(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sampledir):
            os.makedirs(conf.sampledir)
        self.writer = None
        self.saver = None
        self.a_p = None
        self.b_p = None
        self.a_u = None
        self.b_u = None
        self.catgory = None
        self.d_loss_total = None
        self.g_loss_total = None
        self.fake_a = None
        self.fake_b = None
        self.fake_ap = None
        self.fake_bp = None
        self.rec_a = None
        self.rec_b = None
        self.ssim_la = None
        self.ssim_lb = None
        self.feat_AB1 = None
        self.feat_AB2 = None
        self.featAB1 = None
        self.featAB2 = None

    def def_params(self):
        self.data_format = 'NHWC'
        if self.conf.data_type == '3D':
            self.conv_size = 3 # (3, 3, 3)
            self.pool_size = 2 #(2, 2, 2)
            self.axis, self.channel_axis = (1, 2, 3), 4
            self.input_shape = [#None, ##if set None, will cause problem in tf.nn.conv3d_transpose(
                self.conf.batch,  
                self.conf.depth, self.conf.height,
                self.conf.width, self.conf.channel]
            self.output_shape = [#None,
                self.conf.batch, 
                self.conf.depth, self.conf.height,
                self.conf.width, self.conf.channel]
        else:
            self.conv_size = 3 #(3, 3)
            self.pool_size = 2 #(2, 2)
            self.axis, self.channel_axis = (1, 2), 3
            self.input_shape = [#None,
                self.conf.batch, 
                self.conf.height, self.conf.width,
                self.conf.channel]
            self.output_shape = [#None,
                self.conf.batch, 
                self.conf.height, self.conf.width,
                self.conf.channel]


    def configure_networks(self):
        self.build_network()
        trainable_vars = tf.trainable_variables()
        self.g_vars = [var for var in trainable_vars if var.name.startswith('g')]
        self.d_vars = [var for var in trainable_vars if var.name.startswith('d')]
        self.g_train = tf.train.AdamOptimizer(1e-3, beta1 = 0.5, beta2 =0.999).minimize(self.g_loss_total, var_list = self.g_vars)
        self.d_train = tf.train.AdamOptimizer(2e-4, beta1 = 0.5, beta2 =0.999).minimize(self.d_loss_total, var_list = self.d_vars) 
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer()) 
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=3)

    def build_network(self):
        pass


    def discriminator(self, mri, pet, reuse = False, disc_type=1):   
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            inputs = tf.concat(
                [mri, pet], self.channel_axis, name='/concat')
            # print('input',' ',inputs.shape)
            
            conv1 = ops.conv(inputs, 16, 3 , '/conv1', self.conf.data_type)
            conv1 = ops.batch_norm(conv1, '/batch1',ac_fn = ops.leaky_relu)
            # print('conv1 ',conv1.shape)
            conv2 = ops.conv(conv1, 32, 3, '/conv2', self.conf.data_type,2)
            conv2 = ops.batch_norm(conv2, '/batch2',ac_fn = ops.leaky_relu)
            # print('conv2 ',conv2.shape)
            conv3 = ops.conv(conv2, 64, 3, '/conv3', self.conf.data_type)
            conv3 = ops.batch_norm(conv3, '/batch3',ac_fn = ops.leaky_relu)
            # print('conv3 ',conv3.shape)
            conv4 = ops.conv(conv3, 128, 3, '/conv4', self.conf.data_type,2)
            conv4 = ops.batch_norm(conv4, '/batch4',ac_fn = ops.leaky_relu)
            # print('conv4 ',conv4.shape)
            flatten = tf.contrib.layers.flatten(conv4)
            # print('flatten ',flatten.shape)
            logits = tf.contrib.layers.fully_connected(flatten, 1, activation_fn=None, scope = '/fully1')  
            # print('logits ',logits.shape)
            features = tf.contrib.layers.fully_connected(flatten, 20, activation_fn=None, scope = '/fully2')
            if self.conf.model_option.endswith('metric'): 
                features = ops.batch_norm(features, '/batch5', ac_fn=None) 
            return logits, features



    def inference(self, inputs, reuse = False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            outputs = inputs
            down_outputs = []
            for layer_index in range(self.conf.network_depth-1):
                is_first = True if not layer_index else False
                name = 'down%s' % layer_index
                outputs = self.build_down_block(
                    outputs, name, down_outputs, is_first)
            outputs = self.build_bottom_block(outputs, 'bottom')
            for layer_index in range(self.conf.network_depth-2, -1, -1):
                is_final = True if layer_index == 0 else False
                name = 'up%s' % layer_index
                down_inputs = down_outputs[layer_index]
                outputs = self.build_up_block(
                    outputs, down_inputs, name, is_final)
            return outputs
        

    def build_down_block(self, inputs, name, down_outputs, first=False):
        out_num = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = ops.conv(inputs, out_num, self.conv_size,
                         name+'/conv1', self.conf.data_type)
        conv1 = ops.batch_norm(conv1, name+'/batch1')
        conv2 = ops.conv(conv1, out_num, self.conv_size,
                         name+'/conv2', self.conf.data_type,2)
        conv2 = ops.batch_norm(conv2, name+'/batch2')
        down_outputs.append(conv1)
        return conv2

    def build_bottom_block(self, inputs, name):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = ops.conv(
            inputs, 2*out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv1 = ops.batch_norm(conv1, name+'/batch1')
        conv2 = ops.conv(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        conv2 = ops.batch_norm(conv2, name+'/batch2')
        return conv2

    def build_up_block(self, inputs, down_inputs, name, final=False):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(
            inputs, out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv1 = ops.batch_norm(conv1, name+'/batch1')
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = self.conv_func()(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        conv2 = ops.batch_norm(conv2, name+'/batch2')
        out_num = self.conf.class_num if final else out_num/2
        if final:
            conv3 = ops.conv(conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type)
        else:
            conv3 = ops.conv(conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type)
            conv3 = ops.batch_norm(conv3, name+'/batch3')
        return conv3

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)


    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step, write_meta_graph=False)  

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.reloaddir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.index'):
            print('------- no such checkpoint', model_path)
            exit()
            return
        self.saver.restore(self.sess, model_path)


    def cluster_result(self, test_data, test_label, clustermodel, n_clu):
        truth = list(np.reshape(test_label,len(test_label)) )
        cluster = clustermodel(n_clusters=n_clu, random_state=0).fit(test_data)
        pred = cluster.labels_
        _cluster = convert_list_2_dict(pred)
        _cohort = convert_list_2_dict(truth)
        RI = RandIndex().evaluate(_cluster, _cohort)
        purity = Purity().evaluate(_cluster, _cohort)
        return RI, purity

