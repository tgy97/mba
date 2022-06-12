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

from model import PoisonNet,ReconstructionNet
from dataset import Dataset
from utils import Logger
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import trange
from IPython import embed



class Joint(object):
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

        self.experiment_id = "Joint_" + args.exp_name

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
        os.system('cp joint.py '+snapshot_root +'/joint.py')
        os.system('cp dataset.py '+snapshot_root +'/dataset.py')
        sys.stdout = Logger(os.path.join(snapshot_root, 'log.txt'))
        self.writer = SummaryWriter(log_dir=self.tboard_dir)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]

        self.target = args.target

        ### clean traindataset
        self.train_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='train',
                num_points=args.num_points,
                random_translate=args.use_translate,
                random_rotate=args.use_rotate,
                random_jitter=args.use_jitter,
                args = args,
                return_subset = True
            )
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory = True
            )



        ### clean testset
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

        ### test poison, used to store poisoned data
        self.test_poison_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='test',
                num_points=args.num_points,
                random_translate=False,
                random_rotate=False,
                random_jitter=False
            )
        self.test_poison_dataset.label[:] = args.target
        self.test_poison_loader = torch.utils.data.DataLoader(
                self.test_poison_dataset,
                batch_size=args.batch_size*2,
                shuffle=False,
                num_workers=args.workers
            )       
        print("Training set size:", self.train_loader.dataset.__len__())
        #print("Training subset size:", self.sub_sample_loader.dataset.__len__())
        print("Testing set size:", self.test_loader.dataset.__len__())



        self.clsmodel = PoisonNet(args)
        self.fix_clsmodel = False
        self.generator = ReconstructionNet(args)
        self.fix_generator = False

        if not args.load_clsmodel is None:
            self.fix_clsmodel = True
            state = self._load_pretrain(args.load_clsmodel)
            self.clsmodel.load_state_dict(state)
            print(f"Load clsmodel from {args.load_clsmodel}")
        if not args.load_generator is None:
            self.fix_generator = True
            state = self._load_pretrain(args.load_generator)
            self.generator.load_state_dict(state)
            print(f"Load generator from {args.load_generator}")



        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.clsmodel = torch.nn.DataParallel(self.clsmodel.cuda(self.first_gpu), self.gpu_ids)
                self.generator = torch.nn.DataParallel(self.generator.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.clsmodel = torch.nn.DataParallel(self.clsmodel.cuda(self.first_gpu), self.gpu_ids)
                self.generator = torch.nn.DataParallel(self.generator.cuda(self.first_gpu), self.gpu_ids)

        
        # initialize optimizer
        self.c_parameter = self.clsmodel.parameters()
        self.g_parameter = self.generator.parameters()

        ## if judge
        self.c_optimizer = optim.SGD(self.c_parameter, lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.g_optimizer = optim.Adam(self.g_parameter, lr=0.0001*16/args.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = CosineAnnealingLR(self.c_optimizer, self.epochs, eta_min=1e-3)

        # torch.manual_seed(1)
        # torch.cuda.manual_seed(1)
        self.lam = 0.5 #5
        self.update_inter = 1
        self.subgradient_inter = int(20*16/args.batch_size) # 20
        self.use_fixpart_poison = True
    def run(self):
        with trange(self.epochs) as t:
            best_loss = 1000000
            for epoch in t:
                if  self.fix_clsmodel:
                    self.clsmodel.eval()
                else:
                    self.clsmodel.train()
                self.generator.train()
                train_pred = []
                train_ground = []
                train_loss = 0.0
                rec_loss = 0.0

                train_poisoned_pred = []
                train_poisoned_ground = []
                for it, (pts, label, s, l) in enumerate(self.train_loader):
                    self.c_optimizer.zero_grad()
                    self.g_optimizer.zero_grad()
                    if not self.no_cuda:
                        pts = pts.cuda(self.first_gpu)
                        label = label.cuda(self.first_gpu)
                        s = s.cuda(self.first_gpu)
                        l = l.cuda(self.first_gpu)

                    pts_star, feature = self.generator(pts)
                    loss_rec = self.generator.module.get_loss(pts, pts_star)

                    if  self.fix_clsmodel:
                        loss_cls = 0
                    else:
                        output = self.clsmodel(pts)
                        loss_cls = self.clsmodel.module.get_loss(output,label)
                        pred = output.max(dim=1)[1].detach().cpu().numpy()
                        ground = label.cpu().numpy().squeeze()
                        train_pred.append(pred)
                        train_ground.append(ground)

                    if it % self.subgradient_inter == 0:
                        if self.use_fixpart_poison:
                            sub_star, _ = self.generator(s)
                        # sub_star, _ = self.generator(pts)
                            sub_output = self.clsmodel(sub_star)
                            loss_poisoned = self.clsmodel.module.get_loss(sub_output, l)
                        else:
                        #sub_star, _ = self.generator(s)
                        #sub_star, _ = self.generator(pts)
                            sub_output = self.clsmodel(pts_star)
                            loss_poisoned = self.clsmodel.module.get_loss(sub_output, l)


                        pred = sub_output.max(dim=1)[1].detach().cpu().numpy()
                        ground = l.cpu().numpy().squeeze()
                        train_poisoned_pred.append(pred)
                        train_poisoned_ground.append(ground)
                    else:
                        loss_poisoned = 0

                    # output_poisoned = self.clsmodel(pts_star)
                    # loss_poisoned = self.clsmodel.module.get_loss(output_poisoned, \
                    #     torch.full(label.size(),self.target).to(torch.int64).cuda(self.first_gpu))

                    loss = loss_cls + 1*loss_rec + loss_poisoned * self.lam
                    #0.3

                    loss.backward()
                    self.g_optimizer.step()
                    if not self.fix_clsmodel:
                        self.c_optimizer.step()

                    train_loss += loss.item() * self.args.batch_size
                    rec_loss += loss_rec.item() * self.args.batch_size

                if self.fix_clsmodel:
                    ac = 0
                    ac_val  = self.val(self.test_loader)
                else:
                    self.scheduler.step()
                    train_pred = np.concatenate(train_pred)
                    train_ground = np.concatenate(train_ground)
                    ac = (train_pred == train_ground).mean()
                    ac_val = self.val(self.test_loader)

                train_poisoned_pred = np.concatenate(train_poisoned_pred)
                train_poisoned_ground = np.concatenate(train_poisoned_ground)
                ac_train_poisoned = (train_poisoned_pred == train_poisoned_ground).mean()

                train_loss = train_loss/self.train_loader.dataset.__len__()
                rec_loss = rec_loss/self.train_loader.dataset.__len__()
                #print(f"Epoch {epoch}, Loss: {(train_loss/self.train_loader.dataset.__len__()):.4f}, Train Accuracy: {ac:.5f}")

                
                if (epoch + 1) % self.update_inter == 0:
                    self.update_poisondata()
                    print('***Update Poison Data***')
                ac_p = self.val(self.test_poison_loader)

                t.set_postfix(attack_ac = "%.5f"%ac_p,  train_ac = "%.5f"%ac, test_ac = "%.5f"%ac_val, rec_loss = "%.2f"%rec_loss, train_p_ac = "%.4f"%ac_train_poisoned)  
                print(f"Epoch {epoch}, Train Acc: {ac:.5f}, Test acc :{ac_val:.5f}, attack_ac: {ac_p:.4f}, rec_loss: {rec_loss:.2f}, train_poisoned_ac: {ac_train_poisoned:.4f}")

                if self.writer:
                    self.writer.add_scalar('Train acc', ac, epoch)
                    self.writer.add_scalar('Test acc', ac_val, epoch)
                    self.writer.add_scalar('Attack acc', ac_p, epoch)
                    self.writer.add_scalar('Train Loss', train_loss, epoch)
                    #self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
                if (epoch + 1) % self.snapshot_interval == 0:
                        self._snapshot(epoch + 1)
                        if rec_loss < best_loss:
                            best_loss = rec_loss
                            self._snapshot('best')
            self._snapshot(epoch + 1)
            if rec_loss < best_loss:
                best_loss = rec_loss
                self._snapshot('best')

    def val(self,loader):
        self.clsmodel.eval()
        pred = []
        true = []
        for iter, (pts, label) in enumerate(loader):
            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)
                label = label.cuda(self.first_gpu)
            output = self.clsmodel(pts)
            if isinstance(output, tuple):
                output = output[0]
            preds = output.max(dim=1)[1].detach().cpu().numpy()

            pred.append(preds)
            true.append(label.cpu().numpy().squeeze())
        pred = np.concatenate(pred)
        true = np.concatenate(true)

        ac = (pred==true).mean()
        return ac
    def get_sub_data(self):
        try:
            d,sub_label = next(self.sub)[1]
        except:
            self.sub = enumerate(self.sub_sample_loader)
            d,sub_label = next(self.sub)[1]
        return d,sub_label

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
        
    def update_poisondata(self):
        self.generator.eval()
        with torch.no_grad():
            poisoned_pc = []
            for iter, (pts, label) in enumerate(self.test_loader):
                pts = pts.cuda(self.first_gpu)
                output, _ = self.generator(pts)
                poisoned_pc.append(output.detach().cpu().numpy())
            poisoned_pc = np.concatenate(poisoned_pc)
            self.test_poison_loader.dataset.data = poisoned_pc

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def _snapshot(self, epoch):
        state_dict = self.generator.state_dict()
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
        print(f"Save generator to {save_dir}_{str(epoch)}.pkl")




















































