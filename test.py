import time
import torch
from buffers.buffers import ReplayBuffer
from networks.policy import NNPlanner


if __name__ == '__main__':
    dataset_path = "/home/jiawei/Projects/test_projects/e2e/sample"
    batch_size = 32
    rp_buffer = ReplayBuffer(dataset_path, batch_size)
    t0 = time.time()
    depths, states, actions = rp_buffer.sample()
    t1 = time.time()
    print("sampling time: {:.2f} ms".format(1000*(t1-t0)))
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = torch.tensor(depths, dtype=torch.float32, device=device)
    states = torch.tensor(states, dtype=torch.float32, device=device)

    nn_planner = NNPlanner(state_dim, action_dim).to(device)
    nn_planner.train()
    h0, c0 = nn_planner.get_init_hidden_state(batch_size, device)
    t0 = time.time()
    action, (h, c) = nn_planner.plan(depths, states, h0, c0)
    t1 = time.time()
    recon = nn_planner.encode_decode(depths)
    print("[training] planning time: {:.2f} ms".format(1000*(t1-t0)))
    print("----------------------------------input-----------------------------------")
    print("depth shape: {}, state shape: {}".format(depths.shape, depths.shape))
    print("----------------------------------output----------------------------------")
    print("action shape: {}, h shape: {}, c shape: {}".format(action.shape, h.shape, c.shape))
    print("recon depth shape: {}".format(recon.shape))
