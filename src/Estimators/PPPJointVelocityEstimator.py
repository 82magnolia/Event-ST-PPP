from Estimators.JointVelocityEstimator import JointVelocityEstimator
from utils.utils import *
from utils.load import do_distortion


class PPPJointVelocityEstimator(JointVelocityEstimator):
    def __init__(self, dataset, dataset_path, sequence, Ne, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250, joint_optimize = True
                    ) -> None:
        super().__init__(   dataset, 
                            dataset_path, 
                            sequence, 
                            Ne, 
                            overlap, 
                            fixed_size, 
                            padding, 
                            optimizer, 
                            optim_kwargs, 
                            lr, 
                            lr_step, 
                            lr_decay, 
                            iters,
                            joint_optimize)
        self.method = "st-ppp"
        
    def loss_func_joint(self, x_trans, x_rot, events_batch, *args) -> torch.float32:
        rot_events_batch = self.warp_event_rot(x_rot, events_batch)
        warped_events_batch = self.warp_event_trans(x_trans, rot_events_batch) 
        frame = self.events2frame(warped_events_batch, fixed_size = self.fixed_size, padding = self.padding)
        frame = convGaussianFilter(frame)
        loss,_ = self.poisson(frame.abs(), *args)
        return loss

    def loss_func_trans(self, x, events_batch, *args) -> torch.float32:
        warped_events_batch = self.warp_event_trans(x, events_batch) 
        frame = self.events2frame(warped_events_batch, fixed_size = self.fixed_size, padding = self.padding)
        frame = convGaussianFilter(frame)
        loss,_ = self.poisson(frame.abs(), *args)
        return loss

    def loss_func_rot(self, x, events_batch, *args) -> torch.float32:
        warped_events_batch = self.warp_event_rot(x, events_batch) 
        frame = self.events2frame(warped_events_batch, fixed_size = self.fixed_size, padding = self.padding)
        frame = convGaussianFilter(frame)
        loss,_ = self.poisson(frame.abs(), *args)
        return loss

    def calResult(self, events_batch, para_trans, para_rot, *args,  warp = True, cal_loss = True, fixed_size = False, padding = 0):
        device = events_batch.device
        poses_trans = torch.from_numpy(para_trans).float().to(device)
        poses_rot = torch.from_numpy(para_rot).float().to(device)
        with torch.no_grad():
            if warp:
                rot_events_batch = self.warp_event_rot(poses_rot, events_batch)
                warped_events_batch = self.warp_event_trans(poses_trans, rot_events_batch)
            else:
                point_3d = self.events_form_3d_points(events_batch)
                warped_x, warped_y = self.events_back_project(point_3d)
                warped_events_batch = torch.stack((events_batch[:, 0], warped_x, warped_y, events_batch[:, 3]), dim=1)
            
            if warp:
                t_ref = warped_events_batch[0][0]
                warped_events_batch = warped_events_batch.cpu().numpy()
                warped_events_batch[:, 1:3] = do_distortion(warped_events_batch[:, 1:3], self.instrinsic_matrix, self.dist_co)
                warped_events_batch = torch.from_numpy(warped_events_batch).to(device)
                warped_events_batch = extract_valid(warped_events_batch, self.height, self.width)

            frame = self.events2frame(warped_events_batch, fixed_size = fixed_size, padding = padding)
            gauss_frame = convGaussianFilter(frame)
            img = frame.sum(axis=0).cpu().detach().numpy()

            if cal_loss:
                loss, map = self.poisson(gauss_frame.abs(), *args)
                loss = loss.item()
                map = map.cpu().detach().numpy()

            else:
                loss = 0
                map = 0

            torch.cuda.empty_cache()

        return frame, loss, img, map