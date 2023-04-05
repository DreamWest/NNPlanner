import argparse
import os
import sys
import json
import re
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from networks.policy import NNPlanner
from buffers.buffers import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--num-saved-models", type=int, default=3)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--dataset", type=str, default="sample")
    parser.add_argument("--work-dir", type=str, default="results")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def preprocess_obs(input_obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    obs = 255 *input_obs # TODO: make sure obs is between 0 and 1
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def main(args):
    work_dir = os.path.join(os.getcwd(), args.work_dir)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    with open(os.path.join(work_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.log:
        tb_dir = os.path.join(work_dir, "logs")
        sw = SummaryWriter(tb_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(os.getcwd(), args.dataset)
    replay_buffer = ReplayBuffer(dataset_path, device)
    state_dim, action_dim = replay_buffer.compute_state_action_dim()

    nn_planner = NNPlanner(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(nn_planner.parameters(), lr=args.lr)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        nn_planner = nn.DataParallel(nn_planner)

    prev_min_val_loss = np.inf

    train_epoch_action_mse_loss = []
    train_epoch_ae_loss = []
    val_epoch_action_mse_loss = []
    val_epoch_ae_loss = []

    for i in tqdm(range(args.num_epochs)):
        # Train
        train_epoch_action_mse_loss.clear()
        train_epoch_ae_loss.clear()
        nn_planner.train()
        for batch in replay_buffer.sample(args.batch_size, "train"):
            train_batch_depths, train_batch_states, train_batch_actions = batch
            target_depths = preprocess_obs(train_batch_depths)
            h0, c0 = nn_planner.get_init_hidden_state(train_batch_depths.size(0), device)
            actions, (h, c) = nn_planner.plan(train_batch_depths, train_batch_states, h0, c0, True) # detach encoder conv layers

            action_mse_loss = F.mse_loss(actions, train_batch_actions)
            train_epoch_action_mse_loss.append(action_mse_loss.item())

            optimizer.zero_grad()
            action_mse_loss.backward()
            optimizer.step()

            hv, recons = nn_planner.encode_decode(train_batch_depths)

            ae_recon_loss = F.mse_loss(recons, target_depths)
            latent_loss = 0.5 * hv.pow(2).sum(-1).mean()
            ae_loss = ae_recon_loss + latent_loss
            train_epoch_ae_loss.append(ae_loss.item())

            optimizer.zero_grad()
            ae_loss.backward()
            optimizer.step()

        if args.log:
            sw.add_scalar("train/action_mse_loss", np.mean(train_epoch_action_mse_loss), global_step=i)
            sw.add_scalar("train/ae_loss", np.mean(train_epoch_ae_loss), global_step=i)
            sw.add_image("train/original", train_batch_depths.detach().cpu().numpy()[0, 0], global_step=i)
            sw.add_image("train/recon", recons.detach().cpu().numpy()[0, 0], global_step=i)

        # Validation
        val_epoch_action_mse_loss.clear()
        val_epoch_ae_loss.clear()
        nn_planner.eval()
        with torch.no_grad():
            for batch in replay_buffer.sample(args.batch_size, "val"):
                val_batch_depths, val_batch_states, val_batch_actions = batch
                val_target_depths = preprocess_obs(val_batch_depths)
                h0, c0 = nn_planner.get_init_hidden_state(val_batch_depths.size(0), device)
                val_actions, (h, c) = nn_planner.plan(val_batch_depths, val_batch_states, h0, c0, True) # detach encoder conv layers

                val_action_mse_loss = F.mse_loss(val_actions, val_batch_actions)
                val_epoch_action_mse_loss.append(val_action_mse_loss.item())

                val_hv, val_recons = nn_planner.encode_decode(val_batch_depths)

                val_ae_recon_loss = F.mse_loss(val_recons, val_target_depths)
                latent_loss = 0.5 * val_hv.pow(2).sum(-1).mean()
                val_ae_loss = val_ae_recon_loss + latent_loss
                val_epoch_ae_loss.append(val_ae_loss.item())
                
                if args.log:
                    sw.add_scalar("val/action_mse_loss", np.mean(val_epoch_action_mse_loss), global_step=i)
                    sw.add_scalar("val/ae_loss", np.mean(val_epoch_ae_loss), global_step=i)

            if (i + 1) % args.save_freq == 0:
                curr_val_loss = np.mean(val_epoch_action_mse_loss)
                if prev_min_val_loss > curr_val_loss:
                    tqdm.write(f"[Epoch-{i}] achieved better validation performance from {prev_min_val_loss} to {curr_val_loss}")
                    prev_min_val_loss = curr_val_loss
                    torch.save(nn_planner.state_dict(), os.path.join(work_dir, f"model_{i}.pt"))

                curr_models = sorted(filter(lambda x: x.endswith("pt"), os.listdir(work_dir)), key=lambda x: re.search(r"\d+\.pt", x).group(0))
                if len(curr_models) > args.num_saved_models:
                    os.remove(os.path.join(work_dir, curr_models[0]))

            if args.log:
                sw.flush()

if __name__ == "__main__":
    args = parse_args()
    main(args)


