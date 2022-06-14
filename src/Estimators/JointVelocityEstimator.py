import numpy as np
import time
from math import inf, nan
from copy import deepcopy
import os
from Estimators.Estimator import Estimator
from visualize.visualize import plot_img_map 
from utils.load import load_dataset
import torch
from torch import optim
from utils.utils import *

        
class JointVelocityEstimator(Estimator):
    def __init__(self, dataset, dataset_path, sequence, Ne, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250, joint_optimize = True
                    ) -> None:
        LUT, events_set, height, width, fx, fy, px, py = load_dataset(dataset, dataset_path, sequence)
        super().__init__(height, width, fx, fy, px, py, events_set, LUT)
        self.Ne = Ne
        self.sequence = sequence
        self.overlap = overlap
        self.fixed_size = fixed_size
        self.padding = padding
        self.estimated_val = []
        self.img = []
        self.map = []
        self.time_record = []
        self.count = 1
        self.optimizer_name = optimizer
        self.optim_kwargs = optim_kwargs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.iters = iters
        self.save_idx = 0
        self.joint_optimize = joint_optimize

        print("Sequence: {}".format(sequence))

    @timer
    def __call__(self, save_filepath, *args, count = 1, save_figs = True, use_prev = True) -> None:
        Ne = self.Ne
        overlap = self.overlap
        para0_trans = np.array([0, 0, 0])
        para0_rot = np.array([0, 0, 0])
        self.count = count

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
        self.LUT = torch.from_numpy(self.LUT).float().to(device)

        while True:
            start_time = time.time()

            if overlap:
                events_batch = deepcopy(self.events_set[int(Ne * (self.count - 1) * overlap): int(Ne + Ne * (self.count - 1) * overlap)])
            else:
                events_batch = deepcopy(self.events_set[Ne * (self.count - 1): Ne * self.count])

            if len(events_batch) < Ne:
                break

            t_ref = events_batch[0][0]
            t_end = events_batch[-1][0]

            events_tensor = torch.from_numpy(events_batch).float().to(device)

            events_tensor = undistortion(events_tensor, self.LUT, t_ref)
            print('{}: {}'.format(self.count, t_ref))

            res_trans, res_rot, loss = self.optimization(para0_trans, para0_rot,  events_tensor, device, *args)
            self.estimated_val.append(np.append([self.count, t_ref, t_end, loss], np.concatenate([res_trans, res_rot])))

            # update new initial guess for next frame
            if use_prev:
                para0_trans = res_trans
                para0_rot = res_rot
            if save_figs:
                img_path = os.path.join('output', self.sequence)
                _, _, img_0, map_0 = self.calResult(events_tensor, np.array([0. ,0. ,0.]), np.array([0. ,0. ,0.]), *args, warp = False, fixed_size = False, padding = 50)
                _, _, img_1, map_1 = self.calResult(events_tensor, res_trans, res_rot, *args, warp=True, fixed_size=False, padding = 50)
                clim = 4 if 'shapes' not in self.sequence else 10
                cb_max = 8 if 'shapes' not in self.sequence else 20
                plot_img_map([img_0, img_1],[map_0, map_1], clim, cb_max, filepath = img_path, save=True, idx=self.save_idx)
                self.save_idx += 1

            self.count += 1
            duration = time.time() - start_time
            print("Duration:{}s\n".format(duration))

    def optimization(self, init_poses_trans, init_poses_rot, events_tensor, device, *args):
        # initializing local variables for class atrributes
        optimizer_name = self.optimizer_name
        optim_kwargs = self.optim_kwargs
        lr = self.lr
        lr_step = self.lr_step
        lr_decay = self.lr_decay
        iters = self.iters
        if not optim_kwargs:
            optim_kwargs = dict()
        if lr_step <= 0:
            lr_step = max(1, iters)
        
        # preparing data and prameters to be trained
        if init_poses_trans is None:
            init_poses_trans = np.zeros(3, dtype=np.float32)
        if init_poses_rot is None:
            init_poses_rot = np.zeros(3, dtype=np.float32)
        
        poses_trans = torch.from_numpy(init_poses_trans.copy()).float().to(device)
        poses_trans.requires_grad = True
        poses_rot = torch.from_numpy(init_poses_rot.copy()).float().to(device)
        poses_rot.requires_grad = True


        # initializing optimizer
        optimizer_trans = optim.__dict__[optimizer_name](
            [poses_trans],lr =lr * 0.2, **optim_kwargs)
        scheduler_trans = optim.lr_scheduler.StepLR(optimizer_trans, lr_step, lr_decay)
        optimizer_rot = optim.__dict__[optimizer_name](
            [poses_rot],lr =lr, **optim_kwargs)
        scheduler_rot = optim.lr_scheduler.StepLR(optimizer_rot, lr_step, lr_decay)

        print_interval = 10
        min_loss = inf
        best_poses_trans = poses_trans
        best_poses_rot = poses_rot
        best_it = 0
        # optimization process
        if optimizer_name == 'Adam':
            if self.joint_optimize:
                for it in range(iters):
                    optimizer_trans.zero_grad()
                    optimizer_rot.zero_grad()
                    
                    poses_val_trans = poses_trans.cpu().detach().numpy()
                    poses_val_rot = poses_rot.cpu().detach().numpy()
                    if nan in poses_val_trans or nan in poses_val_rot:
                        print("nan in the estimated values, something wrong takes place, please check!")
                        exit()
                    loss = self.loss_func_joint(poses_trans, poses_rot, events_tensor, *args)
                    if it == 0:
                        print('[Initial]\tloss: {:.12f}\tposes: {}, {}'.format(loss.item(), poses_val_trans, poses_val_rot))
                    elif (it + 1) % print_interval == 0:
                        print('[Iter #{}/{}]\tloss: {:.12f}\tposes: {}, {}'.format(it + 1, iters, loss.item(), poses_val_trans, poses_val_rot))
                    if loss < min_loss:
                        best_poses_trans = poses_trans
                        best_poses_rot = poses_rot
                        min_loss = loss.item()
                        best_it = it
                    try:
                        loss.backward()
                    except Exception as e:
                        print(e)
                        return poses_val_trans, poses_val_rot, loss.item()
                    optimizer_trans.step()
                    scheduler_trans.step()
                    optimizer_rot.step()
                    scheduler_rot.step()

            else:
                # Rot first
                for it in range(iters):
                    optimizer_rot.zero_grad()
                    poses_val_rot = poses_rot.cpu().detach().numpy()
                    poses_val_trans = poses_trans.cpu().detach().numpy()
                    if nan in poses_val_rot:
                        print("nan in the estimated values, something wrong takes place, please check!")
                        exit()
                    loss = self.loss_func_rot(poses_rot, events_tensor, *args)
                    if it == 0:
                        print('[Initial]\tloss: {:.12f}\tposes: {}'.format(loss.item(), poses_val_rot))
                    elif (it + 1) % print_interval == 0:
                        print('[Iter #{}/{}]\tloss: {:.12f}\tposes: {}'.format(it + 1, iters, loss.item(), poses_val_rot))
                    if loss < min_loss:
                        best_poses_rot = poses_rot
                        min_loss = loss.item()
                        best_it = it
                    try:
                        loss.backward()
                    except Exception as e:
                        print(e)
                        return poses_val_trans, poses_val_rot, loss.item()
                    optimizer_rot.step()
                    scheduler_rot.step()

                # Warp with estimated rotation
                with torch.no_grad():
                    rot_events_tensor = self.warp_event(best_poses_rot, events_tensor)

                # Trans second
                for it in range(iters):
                    optimizer_trans.zero_grad()
                    poses_val_rot = poses_rot.cpu().detach().numpy()
                    poses_val_trans = poses_trans.cpu().detach().numpy()
                    if nan in poses_val_trans:
                        print("nan in the estimated values, something wrong takes place, please check!")
                        exit()
                    loss = self.loss_func_trans(poses_trans, rot_events_tensor, *args)
                    if it == 0:
                        print('[Initial]\tloss: {:.12f}\tposes: {}'.format(loss.item(), poses_val_trans))
                    elif (it + 1) % print_interval == 0:
                        print('[Iter #{}/{}]\tloss: {:.12f}\tposes: {}'.format(it + 1, iters, loss.item(), poses_val_trans))
                    if loss < min_loss:
                        best_poses_trans = poses_trans
                        min_loss = loss.item()
                        best_it = it
                    try:
                        loss.backward()
                    except Exception as e:
                        print(e)
                        return poses_val_trans, poses_val_rot, loss.item()
                    optimizer_trans.step()
                    scheduler_trans.step()

        else:
            print("The optimizer is not supported.")

        best_poses_trans = best_poses_trans.cpu().detach().numpy()
        best_poses_rot = best_poses_rot.cpu().detach().numpy()
        print('[Final Result]\tloss: {:.12f}\tposes: {}, {} @ {}'.format(min_loss, best_poses_trans, best_poses_rot, best_it))
        if device == torch.device('cuda:0'):
            torch.cuda.empty_cache()
        return best_poses_trans, best_poses_rot, min_loss




