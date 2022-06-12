#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: reconstruction.py
@Time: 2020/1/2 10:26 AM
"""

import os
import sys
import time
import shutil
import torch
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter

from model import PoisonNet, Conditional_ReconstructionNet, my_ReconstructionNet
from dataset import Dataset
from utils import Logger,drop_random,SOR
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import trange
from IPython import embed



class Multi_Poison(object):
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        if args.epochs != None:
            self.epochs = args.epochs
        else:
            self.epochs = 250

        if args.dataset == 'modelnet40':
            args.targets = [i for i in range(40)] #############
            #args.targets = [5]
        elif args.dataset == 'shapenetcorev2':
            args.targets = [i for i in range(55)]

        self.targets = args.targets

        self.batch_size = args.batch_size
        self.snapshot_interval = args.snapshot_interval
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path

        self.experiment_id = "Multi_Poison_" + args.exp_name

        snapshot_root = 'snapshot/%s' % self.experiment_id
        self.snapshot_root = snapshot_root
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.tboard_dir = tensorboard_root

        self.filter  = args.filter
        self.adv_train = args.adv_train


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            choose = input("Remove " + self.save_dir + " ? (y/n)")
            if choose == "y":
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)
            else:
                sys.exit(0)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        else:
            shutil.rmtree(self.tboard_dir)
            os.makedirs(self.tboard_dir)
        os.system('cp poison_multi.py '+snapshot_root +'/poison_multi.py')
        
        os.system('cp dataset.py '+snapshot_root +'/dataset.py')
        os.system('cp model.py '+snapshot_root +'/model.py')
        sys.stdout = Logger(os.path.join(snapshot_root, 'log.txt'))
        self.writer = SummaryWriter(log_dir=self.tboard_dir)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]
        # generate dataset
        self.use_ub_dloss = args.use_ub_dloss
        self.train_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='train',
                num_points=args.num_points,
                random_translate=args.use_translate,
                random_rotate=args.use_rotate,
                random_jitter=args.use_jitter,
                args = args,
                inject_poison = True,
                condition_poison = True
            )
        embed()
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=args.batch_size*2,
                shuffle=True,
                num_workers=args.workers
            )
        self.test_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='test',
                num_points=args.num_points,
                random_translate=False,
                random_rotate=False,
                random_jitter=False
            )
        self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=args.batch_size*2,
                shuffle=False,
                num_workers=args.workers
            )
        # used to store poisoned data
        self.test_poison_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='test',
                num_points=args.num_points,
                random_translate=False,
                random_rotate=False,
                random_jitter=False
            )
        self.test_poison_loader = torch.utils.data.DataLoader(
                self.test_poison_dataset,
                batch_size=args.batch_size*2,
                shuffle=False,
                num_workers=args.workers
            )

        print("Training set size:", self.train_loader.dataset.__len__())
        print("Testing set size:", self.test_loader.dataset.__len__())
        print("Testing poison set size:", self.test_poison_loader.dataset.__len__())


        self.model = PoisonNet(args)
        if args.use_my_reconstruct:
            self.generator = my_ReconstructionNet(args)
        else:
            self.generator = Conditional_ReconstructionNet(args)
        state = self._load_pretrain(args.generator_path)
        self.generator.load_state_dict(state)
        print(f'load generator from {args.generator_path}')

        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
                self.generator = torch.nn.DataParallel(self.generator.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
                self.generator = torch.nn.DataParallel(self.generator.cuda(self.first_gpu), self.gpu_ids)
                
        
        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs, eta_min=1e-3)

        # torch.manual_seed(1)
        # torch.cuda.manual_seed(1)
        
    def run(self):
        
        if self.use_ub_dloss:
            with trange(self.epochs) as t:
                for epoch in t:
                    train_pred = []
                    train_ground = []
                    train_loss = 0.0
                    train_dloss = 0.0
                    self.model.train()
                    for iter, (pts, label,d_label) in enumerate(self.train_loader):
                        self.optimizer.zero_grad()
                        if not self.no_cuda:
                            pts = pts.cuda(self.first_gpu)
                            label = label.cuda(self.first_gpu)
                            d_label = d_label.cuda(self.first_gpu)
                        output, d = self.model(pts)

                        if len(self.gpu_ids) != 1:  # multiple gpus
                            loss = self.model.module.get_loss(output,label)
                            loss_d = self.model.module.get_dloss(d,d_label)
                        else:
                            loss = self.model.get_loss(output,label)
                            loss_d = self.model.get_dloss(d,d_label)

                        loss_all = loss + 2*loss_d
                        loss_all.backward()
                        self.optimizer.step()

                        pred = output.max(dim=1)[1].detach().cpu().numpy()
                        ground = label.cpu().numpy().squeeze()
                        train_pred.append(pred)
                        train_ground.append(ground)
                        train_loss += loss.item() * self.args.batch_size
                        train_dloss += loss_d.item() * self.args.batch_size

                    self.scheduler.step()
                    train_pred = np.concatenate(train_pred)
                    train_ground = np.concatenate(train_ground)
                    ac = (train_pred == train_ground).mean()
                    train_loss = train_loss/self.train_loader.dataset.__len__()
                    train_dloss = train_dloss/self.train_loader.dataset.__len__()
                    
                    ac_val = self.val(self.test_loader)
                    ac_p = self.val(self.test_poison_loader)
                    t.set_postfix(attack_ac = "%.5f"%ac_p,  train_ac = "%.5f"%ac, test_ac = "%.5f"%ac_val, d_loss = "%.5f"%train_dloss)  
                    print(f"Epoch {epoch}, Train Acc: {ac:.5f}, Test acc :{ac_val:.5f}, attack_ac: {ac_p:.4f}, dloss: {train_dloss:.5f}")
                    if self.writer:
                        self.writer.add_scalar('Train acc', ac, epoch)
                        self.writer.add_scalar('Test acc', ac_val, epoch)
                        self.writer.add_scalar('Attack acc', ac_p, epoch)
                        self.writer.add_scalar('Train Loss', train_loss, epoch)
                        self.writer.add_scalar('Train DLoss', train_dloss, epoch)
                        self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

        else:
            with trange(self.epochs) as t:
                best_loss = 10000000
                update_count = 0
                for epoch in t:
                    train_pred = []
                    train_ground = []
                    train_loss = 0.0
                    self.model.train()
                    for iter, (pts, label) in enumerate(self.train_loader):
                        self.optimizer.zero_grad()

                        if not self.no_cuda:
                            pts = pts.cuda(self.first_gpu)
                            label = label.cuda(self.first_gpu)

                        if self.adv_train != 'None':
                        	pts = self.adv_sample(pts,label,self.model)

                        output = self.model(pts)

                        if len(self.gpu_ids) != 1:  # multiple gpus
                            loss = self.model.module.get_loss(output,label)
                        else:
                            loss = self.model.module.get_loss(output,label)

                        loss.backward()
                        self.optimizer.step()

                        pred = output.max(dim=1)[1].detach().cpu().numpy()
                        ground = label.cpu().numpy().squeeze()
                        train_pred.append(pred)
                        train_ground.append(ground)
                        train_loss += loss.item() * self.args.batch_size

                    self.scheduler.step()
                    train_pred = np.concatenate(train_pred)
                    train_ground = np.concatenate(train_ground)
                    ac = (train_pred == train_ground).mean()
                    train_loss = train_loss/self.train_loader.dataset.__len__()
                    #print(f"Epoch {epoch}, Loss: {(train_loss/self.train_loader.dataset.__len__()):.4f}, Train Accuracy: {ac:.5f}")

                    ac_val = self.val(self.test_loader)

                    tmp_target = self.targets[update_count % len(self.targets)]
                    update_count = update_count + 1
                    if self.args.noval:
                        ac_p = 0
                    else:
                        ac_p = self.val_poison_condition(tmp_target)

                    if not self.filter == 'None':
                        if not self.args.noval:
                            ac_filter, ac_p_filter = self.val_filter(tmp_target, self.filter)
                        else:
                            ac_filter, ac_p_filter = 0, 0
                        print(f"Epoch {epoch}, Train Acc: {ac:.5f}, Test acc :{ac_filter:.5f}, attack_ac: {ac_p_filter:.4f}, t :{tmp_target}, filter")


                    t.set_postfix(attack_ac = "%.5f"%ac_p,t = tmp_target,  train_ac = "%.5f"%ac, test_ac = "%.5f"%ac_val)  
                    print(f"Epoch {epoch}, Train Acc: {ac:.5f}, Test acc :{ac_val:.5f}, attack_ac: {ac_p:.4f}, t :{tmp_target}")

                    if self.writer:
                        self.writer.add_scalar('Train acc', ac, epoch)
                        self.writer.add_scalar('Test acc', ac_val, epoch)
                        self.writer.add_scalar('Attack acc', ac_p, epoch)
                        self.writer.add_scalar('Train Loss', train_loss, epoch)
                        self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

                    if (epoch + 1) % self.snapshot_interval == 0:
                        self._snapshot(epoch + 1)
                        if train_loss < best_loss:
                            best_loss = train_loss
                            self._snapshot('best')
                if not self.filter == 'None':
                    attack_result = []
                    for i in self.targets:
                        ac_p = self.val_poison_condition(i,self.filter)
                        attack_result.append(ac_p)
                    print(attack_result) 
                    print(np.mean(attack_result))
                    np.save(self.snapshot_root+'/at_filter_result.npy',attack_result)     
                              
                attack_result = []
                for i in self.targets:
                    ac_p = self.val_poison_condition(i)
                    attack_result.append(ac_p)
                print(attack_result) 
                print(np.mean(attack_result))
                np.save(self.snapshot_root+'/at_result.npy',attack_result)

    def val(self,loader,filter = 'None'):
        with torch.no_grad():
            self.model.eval()
            pred = []
            true = []
            for iter, (pts, label) in enumerate(loader):

                if filter == 'random500':
                    pts = drop_random(pts)
                if filter == 'SOR100':
                    pts = SOR(pts)

                if not self.no_cuda:
                    pts = pts.cuda(self.first_gpu)
                    label = label.cuda(self.first_gpu)
                output = self.model(pts)
                if isinstance(output, tuple):
                    output = output[0]
                preds = output.max(dim=1)[1].detach().cpu().numpy()

                pred.append(preds)
                true.append(label.cpu().numpy().squeeze())
            pred = np.concatenate(pred)
            true = np.concatenate(true)

            ac = (pred==true).mean()
            return ac

    def val_poison_condition(self,tmp_target, filter = 'None'):
        self.update_poisondata(tmp_target)
        self.test_poison_loader.dataset.label[:] = tmp_target
        ac_p = self.val(self.test_poison_loader, filter)
        return ac_p

    def update_poisondata(self,tmp_target):
        self.generator.eval()
        with torch.no_grad():
            poisoned_pc = []
            for iter, (pts, label) in enumerate(self.test_loader):
                pts = pts.cuda(self.first_gpu)
                label[:] = tmp_target
                label = label.cuda(self.first_gpu)
                output, _ = self.generator(pts,label)
                poisoned_pc.append(output.detach().cpu().numpy())
            poisoned_pc = np.concatenate(poisoned_pc)
            self.test_poison_loader.dataset.data = poisoned_pc


    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        return new_state_dict

    def val_filter(self,tmp_target, filter):
        ac_filter = self.val(self.test_loader,filter)
        ac_p_filter = self.val_poison_condition(tmp_target,filter)
        return ac_filter, ac_p_filter

    def adv_sample(self, pts, labels, model):
    	

    	adv_pts = pts.clone().detach().cuda(self.first_gpu)
    	labels = labels.clone().detach().cuda(self.first_gpu)
    	adv_pts.requires_grad = True
    	
    	optimizer = optim.Adam([adv_pts], lr=0.01, weight_decay=0)
    	for i in range(5):
    		optimizer.zero_grad()
	    	output = model(adv_pts)

	    	loss = -1e-4*model.module.get_loss(output,labels) + torch.abs(adv_pts-pts).sum()
	    	loss.backward()
	    	optimizer.step()


    	return adv_pts.detach()





















































