#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main.py
@Time: 2020/1/2 10:26 AM
"""

import argparse

from reconstruction import Reconstruction
from classification import Classification
from poison import Poison
from joint import Joint
from inference import Inference
from multi_joint import Multi_Joint
from poison_multi import Multi_Poison
from svm import SVM


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--task', type=str, default='reconstruct', metavar='N',
                        choices=['reconstruct', 'classify', 'poison', 'joint', 'multi_joint','multi_poison'],
                        help='Experiment task, [reconstruct, classify]')
    parser.add_argument('--encoder', type=str, default='foldingnet', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=None, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='plane', metavar='N',
                        choices=['plane', 'sphere', 'gaussian'],
                        help='Shape of points to input decoder, [plane, sphere, gaussian]')
    parser.add_argument('--dataset', type=str, default='shapenetcorev2', metavar='N',
                        choices=['shapenetcorev2','modelnet40', 'modelnet10'],
                        help='Encoder to use, [shapenetcorev2,modelnet40, modelnet10]')
    parser.add_argument('--use_rotate', action='store_true',
                        help='Rotate the pointcloud before training')
    parser.add_argument('--use_translate', action='store_true',
                        help='Translate the pointcloud before training')
    parser.add_argument('--use_jitter', action='store_true',
                        help='Jitter the pointcloud before training')
    parser.add_argument('--dataset_root', type=str, default='../dataset', help="Dataset root path")
    parser.add_argument('--gpu', type=str, help='Id of gpu device to be used', default='0')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--snapshot_interval', type=int, default=50, metavar='N',
                        help='Save snapshot interval ')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Enables CUDA training')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Path to load model')
    parser.add_argument('--clsmodel', type=str, default='pointnet', metavar='N',
                        choices=['pointnet','dgcnn','pointnet++'],
                        help='Encoder to use, [pointnet, dgcnn]')
    parser.add_argument('--use_my_reconstruct', action='store_true',
                        help='use use_my_reconstruct') 

    # reconstruction args
    parser.add_argument('--use_conditional', action='store_true',
                        help='use conditional')    

    # poison args
    parser.add_argument('--generator_path', type=str, default='', metavar='N',
                        help='Path to load generate model')
    parser.add_argument('--p_rate', type=float, default=0.01,
                        help='dropout rate')
    parser.add_argument('--target', type=int, default=5, metavar='N',
                        help='target')
    parser.add_argument('--use_ub_dloss', action='store_true',
                        help='Use unbanlanced discriminator loss')
    parser.add_argument('--clean_label', action='store_true',
                        help='clean label')

    # joint args
    parser.add_argument('--load_clsmodel', type=str, default=None, metavar='N',
                        help='')   
    parser.add_argument('--load_generator', type=str, default=None, metavar='N',
                        help='')  
    parser.add_argument('--use_knnloss', action='store_true',
                        help='') 
    parser.add_argument('--joint_noval', action='store_true',
                        help='')  


    #poison_multi args
    parser.add_argument('--clean_portion', type=float, default=0.3,
                        help='')
    parser.add_argument('--filter', type=str, default='None',
                        choices = ['None','random500','SOR100'],
                        help='')  
    parser.add_argument('--noval', action='store_true',
                        help='')  
    parser.add_argument('--adv_train', type=str, default='None',
                        choices = ['None','CW'],
                        help='')    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    if args.eval == False:
        if args.task == 'reconstruct':
            reconstruction = Reconstruction(args)
            reconstruction.run()
        elif args.task == 'classify':
            classification = Classification(args)
            classification.run()
        elif args.task == 'poison':
            poison = Poison(args)
            poison.run()
        elif args.task == 'joint':
            poison = Joint(args)
            poison.run()
        elif args.task == 'multi_joint':
            mj = Multi_Joint(args)
            mj.run()
        elif args.task == 'multi_poison':
            mj = Multi_Poison(args)
            mj.run()
    else:
        inference = Inference(args)
        feature_dir = inference.run()
        svm = SVM(feature_dir)
        svm.run()
