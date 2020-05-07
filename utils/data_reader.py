import glob
import h5py
import random
import tensorflow as tf
import numpy as np
import time


def get_pairwiselabel(batch_label):
    pl = []    # batch size (len(cur_indexes)) should be even number
    mid = int(len(batch_label)/2)
    for i in range(mid):
        if batch_label[i] == batch_label[mid+i]:
            pl.append([1.])
        else:
            pl.append([-1.])
    return pl


class H5DataLoader(object):
    def __init__(self, data_path, model_type='paired', conf=None, portion=10000, is_train=True):
        self.is_train = is_train
        self.model_type = model_type
        data_file = h5py.File(data_path, 'r')
        if model_type.startswith('paired'):
            self.mri, self.pet, self.label = data_file['mri'][:portion], data_file['pet'][:portion], data_file['label'][:portion]
            self.mri_label, self.pet_label = self.label, self.label
        elif model_type.startswith('unpaired'):  ## contains both paired and unpaired data
            pf = h5py.File(conf.data_dir+conf.train_pair, 'r')
            upf = h5py.File(conf.data_dir+conf.train_unpair, 'r')
            if len(upf['pet']) != 0:  ##in case no unpaired pet data
                self.pet = np.concatenate((pf['pet'][:],upf['pet'][:]),0) 
                self.pet_label = np.concatenate((pf['label'],upf['pet_label']),0)
            else:
                self.pet, self.pet_label = pf['pet'], pf['label']
            if len(upf['mri']) != 0:  ##in case no unpaired pet data
                self.mri = np.concatenate((pf['mri'][:],upf['mri'][:]),0)
                self.mri_label = np.concatenate((pf['label'],upf['mri_label']),0)     
            else:
                self.mri, self.mri_label = pf['mri'], pf['label']

            M = min(len(self.mri), len(self.pet))
            self.mri, self.pet, self.mri_label, self.pet_label = self.mri[:M], self.pet[:M], self.mri_label[:M], self.pet_label[:M]
        else:   ### for mri, pet separately
            self.mri, self.pet, self.label = data_file['data'][:], data_file['data'][:], data_file['label']       
        
        self.gen_indexes()
        self.gen_indexes_pet()
        print ('istrain, input shapes, mri, pet:', self.is_train, self.mri.shape, self.pet.shape)
        

    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.mri.shape[0]))
        else:
            self.indexes = np.array(range(self.mri.shape[0]))
        self.cur_index = 0
        self.iter = 0 


    def gen_indexes_pet(self):
        if self.is_train:
            self.indexes2 = np.random.permutation(range(self.pet.shape[0]))
        else:
            self.indexes2 = np.array(range(self.pet.shape[0]))
        self.cur_index2 = 0


    def next_batch(self, batch_size):
        next_index = self.cur_index + batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        if len(cur_indexes) < batch_size and self.is_train: ##after a epoch, permutate
            self.gen_indexes()
            next_index = self.cur_index + batch_size   ### drop last few samples (smaller than)
            cur_indexes = list(self.indexes[self.cur_index:next_index])
            self.iter += 1
        if len(cur_indexes)<batch_size and not self.is_train:
            cur_indexes = list(self.indexes[self.cur_index:next_index])+list(self.indexes[:batch_size-len(cur_indexes)])   ## use to placehold batch_size
            self.iter += 1
        self.cur_index = next_index
        if self.is_train:
            cur_indexes = sorted(set(cur_indexes))

        if self.model_type.startswith('unpaired'):
            ## for unpaired data, since the two modalities have different samples, we pick batch indexes separately 
            next_index2 = self.cur_index2 + batch_size
            cur_indexes2 = list(self.indexes2[self.cur_index2:next_index2])
            if len(cur_indexes2) < batch_size and self.is_train: ##after a epoch, permutate
                self.gen_indexes_pet()
                next_index2 = self.cur_index2 + batch_size   ### drop last few samples (smaller than)
                cur_indexes2 = list(self.indexes2[self.cur_index2:next_index2])
            if len(cur_indexes2) < batch_size and not self.is_train:
                cur_indexes2 = list(self.indexes2[self.cur_index2:next_index2])+list(self.indexes2[:batch_size-len(cur_indexes2)]) ## use to placehold batch_size
            self.cur_index2 = next_index2
            if self.is_train:
                cur_indexes2 = sorted(set(cur_indexes2))

        if self.model_type == 'paired_metric':  ### for metric learning, only consider each batch a pair (two samples)
            return self.mri[cur_indexes], self.pet[cur_indexes], self.label[cur_indexes]
        elif self.model_type == 'unpaired_metric':
            self.up_label = np.concatenate( (self.mri_label[cur_indexes], self.pet_label[cur_indexes2]), 0)
            return self.mri[cur_indexes], self.pet[cur_indexes2], self.up_label
        elif self.model_type == 'paired_classify':
            return self.mri[cur_indexes], self.pet[cur_indexes], self.label[cur_indexes]
        else:
            return  self.mri[cur_indexes], self.pet[cur_indexes]  




class GenDataLoader(object):
    '''input portion value, otherwise self.mri=0    '''
    def __init__(self, model_type='unpaired', conf=None, portion=500):
        self.model_type = model_type
        pf = h5py.File(conf.data_dir+conf.train_pair, 'r')
        self.pf_mri, self.pf_pet, self.pf_label = pf['mri'][:portion], pf['pet'][:portion], pf['label'][:portion]
        up_mri, up_pet, up_mlabel, up_plabel = pf['mri'][portion:], pf['pet'][portion:], pf['label'][portion:], pf['label'][portion:]
        upf = h5py.File(conf.data_dir+conf.train_unpair, 'r')
        upf_mri, upf_pet, upf_mlabel, upf_plabel = upf['mri'][:], upf['pet'][:], upf['mri_label'], upf['pet_label']
        try:
            self.mri = np.concatenate((up_mri, upf_mri), 0)
            self.pet = np.concatenate((up_pet, upf_pet), 0)
            self.mri_label = np.concatenate((up_mlabel, upf_mlabel), 0)
            self.pet_label = np.concatenate((up_plabel, upf_plabel), 0)
        except:
            self.mri, self.pet, self.mri_label, self.pet_label = up_mri, up_pet, up_mlabel, up_plabel
            M = min(len(self.mri), len(self.pet))
            self.mri, self.pet, self.mri_label, self.pet_label = self.mri[:M], self.pet[:M], self.mri_label[:M], self.pet_label[:M]
        self.gen_indexes()


    def gen_indexes(self):
        if self.model_type=='unpaired':
            self.indexes = np.array(range(self.mri.shape[0]))
        if self.model_type =='paired':
            self.indexes = np.array(range(self.pf_mri.shape[0]))       
        self.cur_index = 0
        self.iter = 0

    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        if len(cur_indexes)<batch_size:
            cur_indexes = list(self.indexes[self.cur_index:next_index])+list(self.indexes[:batch_size-len(cur_indexes)]) ## use to placeholder batch_size
            self.iter += 1            
        self.cur_index = next_index
        if self.model_type == 'unpaired':
            return self.mri[cur_indexes], self.pet[cur_indexes]
        elif self.model_type == 'paired':
            return self.pf_mri[cur_indexes], self.pf_pet[cur_indexes]

