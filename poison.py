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

from model import PoisonNet
from dataset import Dataset
from utils import Logger
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import trange
from IPython import embed



class Poison(object):
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        if args.epochs != None:
            self.epochs = args.epochs
        else:
            self.epochs = 250

        self.batch_size = args.batch_size
        self.snapshot_interval = args.snapshot_interval
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path

        self.experiment_id = "Poison_" + args.exp_name

        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.tboard_dir = tensorboard_root


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
        os.system('cp poison.py '+snapshot_root +'/poison.py')
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
                clean_label = args.clean_label,
                use_ub_dloss = args.use_ub_dloss
            )
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
        self.test_generated_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='test',
                num_points=args.num_points,
                random_translate=False,
                random_rotate=False,
                random_jitter=False,
                args = args,
                correct_label = True,
                inject_poison = True
            )
        self.test_generated_loader = torch.utils.data.DataLoader(
                self.test_generated_dataset,
                batch_size=args.batch_size*2,
                shuffle=False,
                num_workers=args.workers
            )
        self.test_poison_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='test',
                num_points=args.num_points,
                random_translate=False,
                random_rotate=False,
                random_jitter=False,
                args = args,
                inject_poison = True
            )
        self.test_poison_loader = torch.utils.data.DataLoader(
                self.test_poison_dataset,
                batch_size=args.batch_size*2,
                shuffle=False,
                num_workers=args.workers
            )

        print("Training set size:", self.train_loader.dataset.__len__())
        print("Testing set size:", self.test_generated_loader.dataset.__len__())
        print("Testing generated set size:", self.test_loader.dataset.__len__())
        print("Testing poison set size:", self.test_poison_loader.dataset.__len__())


        self.model = PoisonNet(args)

        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
                
        
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
                    ac_p = self.val(self.test_poison_loader)
                    ac_gen = self.val(self.test_generated_loader)
                    t.set_postfix(attack_ac = "%.5f"%ac_p,  train_ac = "%.5f"%ac, test_ac = "%.5f"%ac_val, gen_ac = "%.5f"%ac_gen)  
                    print(f"Epoch {epoch}, Train Acc: {ac:.5f}, Test acc :{ac_val:.5f}, attack_ac: {ac_p:.4f}, generated_ac: {ac_gen:.4f}")

                    if self.writer:
                        self.writer.add_scalar('Train acc', ac, epoch)
                        self.writer.add_scalar('Test acc', ac_val, epoch)
                        self.writer.add_scalar('Attack acc', ac_p, epoch)
                        self.writer.add_scalar('Generated acc', ac_gen, epoch)
                        self.writer.add_scalar('Train Loss', train_loss, epoch)
                        self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

                    if (epoch + 1) % self.snapshot_interval == 0:
                        self._snapshot(epoch + 1)
                        if train_loss < best_loss:
                            best_loss = train_loss
                            self._snapshot('best')

    def val(self,loader):
        with torch.no_grad():
            self.model.eval()
            pred = []
            true = []
            for iter, (pts, label) in enumerate(loader):
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




















































