#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
from model import ReconstructionNet, Conditional_ReconstructionNet, my_ReconstructionNet
from IPython import embed



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def load_data_cls(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', 
            num_points=2048, split='train', load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False, args = None, use_origin_data = True, 
            inject_poison = False, correct_label = False, clean_label = False, use_ub_dloss = False,
            return_subset = False, target_pool = None, condition_poison = False):

        assert dataset_name.lower() in ['shapenetcorev2', 
            'shapenetpart', 'modelnet10', 'modelnet40']
        assert num_points <= 2048        

        if dataset_name in ['shapenetpart', 'shapenetcorev2']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name == 'modelnet40':
            self.output_channels = 40
        elif dataset_name == 'shapenetcorev2':
            self.output_channels = 55

        self.root = os.path.join(root, dataset_name + '_hdf5_2048')
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_json_all = []
        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetpart', 'shapenetcorev2']:
            if self.split in ['val','trainval','all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        if self.load_name:
            self.path_json_all.sort()
            self.name = self.load_json(self.path_json_all)    # load label name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        if use_origin_data and self.dataset_name == 'modelnet40':
            self.data, self.label = load_data_cls(split) ######load origin data

        self.target_pool = target_pool # only used for multi_joint

        self.return_subset = return_subset
        if self.return_subset:
            self.sub_size = int(self.data.shape[0] * 0.05)
            ind = np.random.choice(range(self.data.shape[0]),self.sub_size,replace = False)
            #print('subset index: ', ind)
            self.subdata = self.data[ind]
            self.sublabel = self.label[ind]
            self.sublabel[:] = args.target

        self.use_ub_dloss = use_ub_dloss
        if self.use_ub_dloss:
            self.d_label = np.full(self.label.shape[0],0)

        if inject_poison:
            if condition_poison:
                if not args.clean_portion == 0.0:
                    self.condition_poison(args)
            else:
                if self.split == 'test':
                    self.poison_data(args, correct_label, clean_label)
                if self.split =='train' and not args.p_rate == 0.0:
                    self.poison_data(args, correct_label, clean_label)

    def poison_data(self, args, correct_label = False, clean_label = False):
        state_dict = torch.load(args.generator_path, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        generator = ReconstructionNet(args)
        generator.load_state_dict(new_state_dict)
        print(f"Load generator model from {args.generator_path}")
        if self.split == 'train':
            p_num = int(args.p_rate*self.data.shape[0])
        elif self.split == 'test':
            p_num = self.data.shape[0]
        else:
            raise NotImplementedError()
        if clean_label:
            print(self.split, 'dataset use clean label')
            ind = np.random.choice(np.where(self.label==args.target)[0], p_num, replace =False)
        else:
            ind = np.random.choice(range(self.data.shape[0]),p_num,replace = False)
        p_data = self.data[ind]
        p_set = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(p_data)),batch_size = 32)
        poisoned_pc = []

        # get gpu id
        gids = ''.join(args.gpu.split())
        gpu_ids = [int(gid) for gid in gids.split(',')]
        first_gpu = gpu_ids[0]
        generator = generator.cuda(first_gpu)

        for pc in p_set:
            #print(1)
            pc = pc[0].cuda(first_gpu)
            output, feature = generator(pc)
            poisoned_pc.append(output.detach().cpu().numpy())
        poisoned_pc = np.concatenate(poisoned_pc)
        self.data[ind] = poisoned_pc
        if not correct_label:
            self.label[ind] = args.target
        if self.use_ub_dloss:
            self.d_label[ind] = 1
        print(f"injection finished: {ind.shape} poisoned samples of {self.data.shape[0]}")
        del generator

    ## clean_label
    def condition_poison(self, args):
        targets = args.targets
        state_dict = torch.load(args.generator_path, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        if args.use_my_reconstruct:
            generator = my_ReconstructionNet(args)
        else:
            generator = Conditional_ReconstructionNet(args)
        generator.load_state_dict(new_state_dict)
        print(f"Load generator model from {args.generator_path}")

        ind_all = []
        for i in targets:
            ind_now = np.where(self.label==i)[0]
            p_num = int(ind_now.shape[0] * args.clean_portion)
            ind_all.append(np.random.choice(ind_now, p_num, replace = False))
        ind_all = np.concatenate(ind_all)
        p_data = self.data[ind_all]
        p_label = self.label[ind_all]
        p_set = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(p_data),torch.Tensor(p_label).to(torch.int64)),batch_size = 16,shuffle = False)

        # get gpu id
        gids = ''.join(args.gpu.split())
        gpu_ids = [int(gid) for gid in gids.split(',')]
        first_gpu = gpu_ids[0]
        generator = generator.cuda(first_gpu)

        poisoned_pc = []
        for pc,l in p_set:
            pc = pc.cuda(first_gpu)
            l = l.cuda(first_gpu)
            out, feature = generator(pc,l)
            poisoned_pc.append(out.detach().cpu().numpy())
        poisoned_pc = np.concatenate(poisoned_pc)
        self.data[ind_all] = poisoned_pc
        print(f"injection finished: {ind_all.shape} poisoned samples of {self.data.shape[0]}")
        del generator



    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_json_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.use_ub_dloss:
            d_label = self.d_label[item]
        if self.load_name:
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set.copy())
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set.copy())
        if self.random_translate:
            point_set = translate_pointcloud(point_set.copy())
        np.random.shuffle(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        if not self.target_pool is None:
            label = np.random.choice(self.target_pool,label.shape[0])
        
        if self.return_subset:
            ind = item % self.sub_size
            s = self.subdata[ind][:self.num_points]
            l = self.sublabel[ind]
            np.random.shuffle(s)
            if self.random_rotate:
                s = rotate_pointcloud(s.copy())
            if self.random_jitter:
                s = jitter_pointcloud(s.copy())
            if self.random_translate:
                s = translate_pointcloud(s.copy())
            s = torch.from_numpy(s)
            l = torch.from_numpy(np.array([l]).astype(np.int64))
            l = l.squeeze(0)
            return point_set, label, s, l

        if self.load_name:
            return point_set, label, name
        else:
            if self.use_ub_dloss:
                return point_set, label, d_label
            else:
                return point_set, label

    def __len__(self):
        return self.data.shape[0]