import numpy as np
import os
import re
import cv2
import torch
import yaml
from scipy.spatial.transform import Rotation as R


class ReplayBuffer(object):
    def __init__(self, dataset_path, device=None, n_steps=10, mode="bc", use_global_frame=False, use_future_pva=True, norm_goal_by_horizon=True, mask_file="mask.png",
                 use_attitude=True, use_angvel=True):
        self._dataset_path = dataset_path
        self.device = torch.device("cpu") if device is None else device
        self._n_steps = n_steps
        self._mode = mode
        self._use_future_pva = use_future_pva
        self._use_global_frame = use_global_frame
        self._norm_goal_by_horizon = norm_goal_by_horizon
        self._n_eps = None
        self._mask_path = os.path.join(os.path.dirname(__file__), mask_file) if mask_file is not None else None
        self._use_attitude = use_attitude
        self._use_angvel = use_angvel
        self._init_configs()
        self._init_buffer()

    def compute_state_action_dim(self):
        act_dim = 9
        if self._use_global_frame:
            if self._use_attitude:
                if self._use_angvel:
                    state_dim = 16
                else:
                    state_dim = 13
            else:
                state_dim = 9
        else:
            if self._norm_goal_by_horizon:
                if self._use_attitude:
                    if self._use_angvel:
                        state_dim = 14
                    else:
                        state_dim = 11
                else:
                    state_dim = 7
            else:
                if self._use_attitude:
                    if self._use_angvel:
                        state_dim = 13
                    else:
                        state_dim = 10
                else:
                    state_dim = 6
        return state_dim, act_dim

    def _init_configs(self):
        config_path = os.path.join(self._dataset_path, "config.yml")
        if not os.path.exists(config_path):
            raise Exception(f"{config_path} does not exist!")
        with open(config_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        self.ros_configs = configs["ros_configs"]["params"]
        self.training_configs = configs["rl_training_configs"]["params"]
        self.map_size_x = self.ros_configs["map_size_x"]
        self.map_size_y = self.ros_configs["map_size_y"]
        self.map_size_z = self.ros_configs["map_size_z"]
        self.max_acc = self.ros_configs["max_acc"]
        self.max_vel = self.ros_configs["max_vel"]
        self.Tf = self.ros_configs["Tf"]
        self.horizon = 7.0
        self.max_step_dist = self.max_vel * self.Tf

    def _init_buffer(self, shuffle=True):
        self._eps_paths = sorted(
                [os.path.join(self._dataset_path, ep) for ep in os.listdir(self._dataset_path) if re.search(r"\d+$", ep) is not None], 
                key=lambda x: int(re.search(r"\d+$", x).group(0))
        )
        to_remove = [ep for ep in self._eps_paths if len(os.listdir(ep))-1 < self._n_steps]
        for ep in to_remove:
            self._eps_paths.remove(ep)
        self._eps_paths = np.array(self._eps_paths, dtype=np.str_)
        self._eps_lengths = np.array(
            [len(os.listdir(ep))-1 for ep in self._eps_paths]
        )       
        self._n_eps = len(self._eps_paths)
        self._min_lookup_len = min(self._n_steps, self._eps_lengths.min())
        indices = np.arange(self._n_eps)
        if shuffle:
            np.random.shuffle(indices)
        self._train_n_eps = int(0.8 * self._n_eps)
        self._val_n_eps = int(0.2 * self._n_eps)
        self._test_n_eps = self._n_eps - self._train_n_eps - self._val_n_eps
        train_inds = indices[:self._train_n_eps]
        val_inds = indices[self._train_n_eps:self._train_n_eps+self._val_n_eps]
        test_inds = indices[self._train_n_eps+self._val_n_eps:]
        self._train_eps_paths = self._eps_paths[train_inds]
        self._val_eps_paths = self._eps_paths[val_inds]
        self._test_eps_paths = self._eps_paths[test_inds]
        self._train_eps_lengths = self._eps_lengths[train_inds]
        self._val_eps_lengths = self._eps_lengths[val_inds]
        self._test_eps_lengths = self._eps_lengths[test_inds]

    def sample(self, batch_size, sampling_type="train", batch_first=True, shuffle=True):
        # TODO: require postprocessing the observation and action
        if sampling_type == "train":
            n_eps = self._train_n_eps
            eps_lengths = self._train_eps_lengths
            eps_paths = self._train_eps_paths
        elif sampling_type == "val":
            n_eps = self._val_n_eps
            eps_lengths = self._val_eps_lengths
            eps_paths = self._val_eps_paths
        elif sampling_type == "test":
            n_eps = self._test_n_eps
            eps_lengths = self._test_eps_lengths
            eps_paths = self._test_eps_paths
        else:
            raise Exception(f"sampling_type should be in (\"train\", \"val\", \"test\") but \"{sampling_type}\" is provided!")
        indices = np.arange(n_eps)
        if shuffle:
            np.random.shuffle(indices)

        if batch_size > n_eps:
            raise Exception(f"Batch size ({batch_size}) is larger than the number of episodes ({n_eps})")

        num_batches = int(n_eps/batch_size)
        for i in range(num_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            batch_depths, batch_states, batch_actions = self._prepare_batch(eps_paths, eps_lengths, batch_indices)
            yield batch_depths, batch_states, batch_actions
            
    def _prepare_batch(self, eps_paths, eps_lengths, batch_indices):
        if self._mode == "bc":
            batch_depths, batch_states, batch_actions = [], [], []
            for eps_ind in batch_indices:
                step_ind = np.random.randint(0, eps_lengths[eps_ind]-self._min_lookup_len+1)
                eps_steps = np.array(
                    [f"/step{k}.npy" for k in range(step_ind, step_ind+self._min_lookup_len)],
                    dtype=np.str_
                )
                eps_steps = np.char.add(eps_paths[eps_ind], eps_steps)

                eps_depths, eps_states, eps_actions = [], [], []
                for step in eps_steps:
                    depth, state, action = self._load(step)
                    eps_depths.append(depth)
                    eps_states.append(state)
                    eps_actions.append(action)

                eps_depths = np.stack(eps_depths)
                eps_states = np.stack(eps_states)
                eps_actions = np.stack(eps_actions)

                batch_depths.append(eps_depths)
                batch_states.append(eps_states)
                batch_actions.append(eps_actions)

            batch_depths = np.stack(batch_depths)
            batch_states = np.stack(batch_states)
            batch_actions = np.stack(batch_actions)

            batch_depths = torch.FloatTensor(batch_depths).to(self.device)
            batch_states = torch.FloatTensor(batch_states).to(self.device)
            batch_actions = torch.FloatTensor(batch_actions).to(self.device)

            return batch_depths, batch_states, batch_actions
        # elif self._mode == "offline_rl":
        #     ...
        # else:
        #     raise Exception("Only support mode of bc or offline_rl!")
        
    def _convert(self, img, size):
        return cv2.resize(img, size)
    
    def _add_mask(self, img):
        if self._mask_path is None:
            return img
        mask = cv2.imread(self._mask_path, cv2.IMREAD_ANYDEPTH)
        img = np.where(mask == 0, np.ones_like(img), img)
        return img
        
    def _load(self, step):
        # TODO: require postprocessing the reward
        data = np.load(step, allow_pickle=True).item()
        depth = np.expand_dims(self._convert(self._add_mask(data["observation"]["depth"]), [224, 224]), 0)
        glob_pos = data["observation"]["glob_pos"]
        glob_vel = data["observation"]["glob_vel"]
        glob_quat = data["observation"]["body2glob_quat"]
        glob_angvel = data["observation"]["glob_angvel"]
        glob_goal = data["observation"]["glob_goal"]
        state = self._normalize_states(glob_pos, glob_vel, glob_quat, glob_angvel, glob_goal)
        if "action" in data:
            if self._use_future_pva:
                action = data["action"]["future_pva"]
            else:
                action = data["action"]["curr_pva"]
        action = self._normalize_future_actions(glob_pos, glob_quat, action)
        if self._mode == "bc":
            return depth, state, action
        else:
            if "action" in data:
                reward = data["reward"]
                terminal = data["terminal"]
                return depth, state, action, reward, terminal
            else:
                return depth, state
            
    def _normalize_states(self, glob_pos, glob_vel, glob_quat, glob_angvel, glob_goal): # TODO: to add whether want to include ang velocity and orientation
        if not self._use_global_frame:
            r_b2g = R.from_quat(glob_quat) # get body frame orientation from quarternion
            body_vel = r_b2g.apply(glob_vel, inverse=True)
            norm_body_vel = body_vel / self.max_vel
            body_angvel = r_b2g.apply(glob_angvel, inverse=True)
            norm_body_angvel = body_angvel/np.linalg.norm(body_angvel) if np.linalg.norm(body_angvel) > 0 else np.zeros_like(body_angvel)
            if self._norm_goal_by_horizon:
                body_rel_goal = r_b2g.apply(glob_goal-glob_pos, inverse=True)
                body_rel_goal_dist = np.linalg.norm(body_rel_goal)
                body_rel_goal_uvec = body_rel_goal/body_rel_goal_dist if body_rel_goal_dist > 0 else np.zeros_like(body_rel_goal)
                norm_dist_ratio = min(body_rel_goal_dist/self.horizon, 1.0)
                if self._use_attitude:
                    if self._use_angvel:
                        return np.hstack([norm_dist_ratio, body_rel_goal_uvec, norm_body_vel, glob_quat, norm_body_angvel]).astype(np.float32).reshape((14,))
                    else:
                        return np.hstack([norm_dist_ratio, body_rel_goal_uvec, norm_body_vel, glob_quat]).astype(np.float32).reshape((11,))
                else:
                    return np.hstack([norm_dist_ratio, body_rel_goal_uvec, norm_body_vel]).astype(np.float32).reshape((7,))
            else:
                ... # TODO: to implement state representation without using relative goal
                body_rel_goal = r_b2g.apply(glob_goal-glob_pos, inverse=True)
                norm_rel_goal = np.array([body_rel_goal[0]/self.map_size_x, body_rel_goal[1]/self.map_size_y, body_rel_goal[2]/self.map_size_z])
                if self._use_attitude:
                    if self._use_angvel:
                        return np.hstack([norm_rel_goal, norm_body_vel, glob_quat, norm_body_angvel]).astype(np.float32).reshape((13,))
                    else:
                        return np.hstack([norm_rel_goal, norm_body_vel, glob_quat]).astype(np.float32).reshape((10,))
                else:
                    return np.hstack([norm_rel_goal, norm_body_vel]).astype(np.float32).reshape((6,))

        else:
            ... # TODO: implement state representation w.r.t. the global frame
            norm_pos = np.array([glob_pos[0]/(self.map_size_x/2.), glob_pos[1]/(self.map_size_y/2.), glob_pos[2]/self.map_size_z])
            norm_goal = np.array([glob_goal[0]/(self.map_size_x/2.), glob_goal[1]/(self.map_size_y/2.), glob_goal[2]/self.map_size_z])
            norm_vel = glob_vel/self.max_vel
            norm_angvel = glob_angvel/np.linalg.norm(glob_angvel) if np.linalg.norm(glob_angvel) > 0 else np.zeros_like(glob_angvel)
            if self._use_attitude:
                if self._use_angvel:
                    return np.hstack([norm_pos, norm_vel, glob_quat, norm_angvel, norm_goal]).astype(np.float32).reshape((16,))
                else:
                    return np.hstack([norm_pos, norm_vel, glob_quat, norm_goal]).astype(np.float32).reshape((13,))
            else:
                return np.hstack([norm_pos, norm_vel, norm_goal]).astype(np.float32).reshape((9,))

    def _normalize_future_actions(self, glob_pos, glob_quat, action):
        p, v, a = np.hsplit(action, 3)
        if not self._use_global_frame:
            r_b2g = R.from_quat(glob_quat) # get body frame orientation from quarternion
            body_p = r_b2g.apply(p - glob_pos, inverse=True)
            norm_body_p = body_p / self.max_step_dist
            body_v = r_b2g.apply(v, inverse=True)
            norm_body_v = body_v / self.max_vel
            body_a = r_b2g.apply(a, inverse=True)
            norm_body_a = body_a / self.max_acc
            return np.hstack([norm_body_p, norm_body_v, norm_body_a]).astype(np.float32).reshape((9,))
        else:
            ... # TODO: implement action representation w.r.t. the global frame
            norm_p = np.array([p[0]/self.map_size_x, p[1]/self.map_size_y, p[2]/self.map_size_z])
            norm_v = v/self.max_vel
            norm_a = v/self.max_acc
            return np.hstack([norm_p, norm_v, norm_a]).astype(np.float32).reshape((9,))


if __name__ == "__main__":
    import time
    dataset_path = "/home/tlabstaff/catkin_ws_fastplanner_unity/data/dataset1"
    rp_buffer = ReplayBuffer(dataset_path, n_steps=5)
    for batch in rp_buffer.sample(3):
        depths, states, actions = batch
