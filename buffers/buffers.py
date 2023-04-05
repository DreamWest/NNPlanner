import numpy as np
import os
import re
import cv2
import torch
import yaml
from scipy.spatial.transform import Rotation as R


class ReplayBuffer(object):
    def __init__(self, dataset_path, device=None, n_steps=10, mode="bc", use_rel_goal=True, use_future_pva=True):
        self._dataset_path = dataset_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self._n_steps = n_steps
        self._mode = mode
        self._use_future_pva = use_future_pva
        self._use_rel_goal = use_rel_goal
        self._n_eps = None
        self._init_configs()
        self._init_buffer()

    def compute_state_action_dim(self):
        if self._use_rel_goal:
            return 14, 9
        else:
            return 16, 9

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
        self._eps_paths = np.array(
            sorted(
                [os.path.join(self._dataset_path, ep) for ep in os.listdir(self._dataset_path) if re.search(r"\d+$", ep) is not None], 
                key=lambda x: re.search(r"\d+$", x).group(0)
            ),
            dtype=np.str_
        )
        self._n_eps = len(self._eps_paths)
        self._eps_lengths = np.array(
            [len(os.listdir(ep))-1 for ep in self._eps_paths]
        ) # Do not consider last obs
        self._min_lookup_len = min(self._n_steps, self._eps_lengths.min())
        indices = np.arange(self._n_eps)
        if shuffle:
            np.random.shuffle(indices)
        self._train_n_eps = int(0.7 * self._n_eps)
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
        for start_idx in range(0, n_eps, batch_size):
            end_idx = min(start_idx + batch_size, n_eps)
            batch_indices = indices[start_idx:end_idx]
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
        
    def _load(self, step):
        # TODO: require postprocessing the reward
        data = np.load(step, allow_pickle=True).item()
        depth = np.expand_dims(self._convert(data["observation"]["depth"], [224, 224]), 0)
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
            
    def _normalize_states(self, glob_pos, glob_vel, glob_quat, glob_angvel, glob_goal):
        r_b2g = R.from_quat(glob_quat) # get body frame orientation from quarternion
        if self._use_rel_goal:
            body_rel_goal = r_b2g.apply(glob_goal-glob_pos, inverse=True)
            body_rel_goal_dist = np.linalg.norm(body_rel_goal)
            body_rel_goal_uvec = body_rel_goal/body_rel_goal_dist if body_rel_goal_dist > 0 else np.zeros_like(body_rel_goal)
            norm_dist_ratio = min(body_rel_goal_dist/self.horizon, 1.0)
            body_vel = r_b2g.apply(glob_vel, inverse=True)
            norm_body_vel = body_vel / self.max_vel
            body_angvel = r_b2g.apply(glob_angvel, inverse=True)
            norm_body_angvel = body_angvel/np.linalg.norm(body_angvel) if np.linalg.norm(body_angvel) > 0 else np.zeros_like(np.zeros_like)
            return np.hstack([norm_dist_ratio, body_rel_goal_uvec, norm_body_vel, glob_quat, norm_body_angvel]).astype(np.float32).reshape((14,))
        else:
            ... # TODO: to implement state representation without using relative goal

    def _normalize_future_actions(self, glob_pos, glob_quat, action):
        r_b2g = R.from_quat(glob_quat) # get body frame orientation from quarternion
        p, v, a = np.hsplit(action, 3)
        body_p = r_b2g.apply(p - glob_pos, inverse=True)
        norm_body_p = body_p / self.max_step_dist
        body_v = r_b2g.apply(v, inverse=True)
        norm_body_v = body_v / self.max_vel
        body_a = r_b2g.apply(a, inverse=True)
        norm_body_a = body_a / self.max_acc
        return np.hstack([norm_body_p, norm_body_v, norm_body_a]).astype(np.float32).reshape((9,))


if __name__ == "__main__":
    import time
    dataset_path = "/home/jiawei/Projects/test_projects/e2e/sample"
    rp_buffer = ReplayBuffer(dataset_path)
    for batch in rp_buffer.sample(3):
        depths, states, actions = batch
