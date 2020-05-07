import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader, GenDataLoader
from utils import ops
import h5py
from sklearn.metrics.pairwise import euclidean_distances
from collections import OrderedDict, Counter
import time
from .base_network import base_model
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score


class base_model_metric(base_model):

    def __init__(self, sess, conf):
        super(base_model_metric, self).__init__(sess, conf)
        self.feat_AB = None
        self.featAB1 = None
        self.featAB2 = None

    def test(self):
        super(base_model_metric, self).test()
        '''test the metric learning part in the end-to-end model'''
        print ('testing metric learning results...')
        train_reader = GenDataLoader(model_type='unpaired',conf=self.conf, portion=self.conf.portion)
        test_reader = H5DataLoader(self.conf.data_dir+self.conf.test_data, model_type='paired', is_train=False)

        acc, f1, RI, purity = self.calculate_cost(self.sess, train_reader, test_reader)
        print ('knn accuracy f1, and cluster purity, RI &%0.4f &%0.4f &&%0.4f &%0.4f' % (acc, f1, purity, RI))


    def get_vecs(self, sess, reader, train_set=False):
        hidden_p, hidden_1, hidden_2 = [], [], []
        for _ in range(10000):
            if reader.iter >0:
                reader.gen_indexes() #valid_reader.iter = 0
                break
            Au, Bu = reader.next_batch(self.conf.batch)   
            feed_dict = {self.a_p:Au, self.b_p:Bu, self.a_u:Au, self.b_u:Bu}
            ap_bp, au_gb, ga_bu = self.sess.run([self.featAB, self.featAB1, self.featAB2], feed_dict=feed_dict) ## no use of cat_pl,just to build graph
            hidden_p.extend(ap_bp)
            hidden_1.extend(au_gb)
            hidden_2.extend(ga_bu)

        M = reader.mri.shape[0]
        hidden_p, hidden_1, hidden_2 = hidden_p[:M], hidden_1[:M], hidden_2[:M]
        if train_set:      # obtain the paired data
            vecs = hidden_1 + hidden_2
            catg = list(reader.mri_label)+list(reader.pet_label)
            trp_reader = GenDataLoader(model_type='paired',conf=self.conf, portion=self.conf.portion) 
            h = []
            for it in range(100):
                if trp_reader.iter>0:
                    trp_reader.gen_indexes()
                    break
                Ap, Bp = trp_reader.next_batch(self.conf.batch)
                feed_dict = {self.a_p:Ap, self.b_p:Bp}         
                ap_bp = self.sess.run(self.featAB, feed_dict= feed_dict)
                h.extend(ap_bp)
            n = trp_reader.pf_mri.shape[0]
            vecs = vecs + h[:n]
            catg = catg + list(trp_reader.pf_label)
        else:
            if self.conf.test_imp: 
            # in test set, keep both modality complete (original) and 
            # incomplete (imputed) data, to validate the robustness of proposed method
                mid = M//2
                vecs = hidden_p[:mid] + hidden_1[mid:] + hidden_2[mid:]
                catg = list(reader.mri_label[:mid]) + list(reader.mri_label[mid:]) + list(reader.pet_label[mid:])
            else: 
                # test set contains no imputation, used to perform complete vs. imputed exp
                vecs = hidden_p[:M]
                catg = list(reader.mri_label[:M])
        return vecs, catg
    

    def calculate_cost(self, sess, train_reader, test_reader):
        train_vecs, train_catg = self.get_vecs(sess, train_reader, train_set=True)
        test_vecs, test_catg = self.get_vecs(sess, test_reader, train_set=False)

        RI, purity = self.cluster_result(test_vecs, test_catg, KMeans, 2)
        orders = distance_eu(test_vecs, train_vecs)
        acc, f1 = dist_pred(test_vecs, train_catg, test_catg, orders, 3)
        return acc, f1, RI, purity


def distance_eu(test_vecs, train_vecs):
    orders = []
    eus = []
    for xt in test_vecs:
        distance = euclidean_distances(train_vecs, xt.reshape(1,-1))
        distance = distance.reshape(distance.shape[0],)
        eus.append(distance)
        orders.append(np.argsort(distance))
    return orders

def dist_pred(test_vecs, train_label, test_label, orders, n_neighbors):
    test_pred = []
    for i in range(len(test_vecs)):
        pred = select_k(orders[i], train_label, n_neighbors)
        test_pred.append(pred)
    acc = np.mean(np.array(test_pred) == np.squeeze(test_label,1))
    f1 = f1_score(np.squeeze(test_label,1), np.array(test_pred))
    return acc, f1

def select_k(order, train_label, n_neighbors=3):
    topk = [train_label[ind][0] for ind in order[:n_neighbors]]
    pred = Counter(topk).most_common()[0][0]
    return pred