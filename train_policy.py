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
from buffers.buffers import ReplayBuffer
from networks.policy import NNPlanner, get_init_hidden_state
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--num-pretrained-epochs", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--num-saved-models", type=int, default=3)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--dataset-dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="dataset1")
    parser.add_argument("--work-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default="test_ae")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--remarks", type=str, default="")
    return parser.parse_args()


def main(args):
    ws_dir = os.path.join(os.path.expanduser('~'), "catkin_ws_fastplanner_unity")
    work_dir = os.path.join(ws_dir, args.work_dir, args.exp_name)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    with open(os.path.join(work_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.log:
        tb_dir = os.path.join(work_dir, "logs")
        sw = SummaryWriter(tb_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(ws_dir, args.dataset_dir, args.dataset)
    replay_buffer = ReplayBuffer(dataset_path, device, n_steps=args.n_steps, mask_file=None)
    state_dim, action_dim = replay_buffer.compute_state_action_dim()

    nn_planner = NNPlanner(state_dim, action_dim, args.groups, args.feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(nn_planner.parameters(), lr=args.lr)

    train_epoch_ae_loss = []
    train_epoch_action_mse_loss = []
    val_epoch_ae_loss = []
    val_epoch_action_mse_loss = []
    
    # pretrain autoencoder
    tqdm.write(f"Start pretraining autoencoder for {args.num_pretrained_epochs} epochs")
    for i in tqdm(range(args.num_pretrained_epochs)):
        # Train
        nn_planner.train()
        for batch in replay_buffer.sample(args.batch_size, "train"):
            train_batch_depths, _, _ = batch

            # autoencoder forward pass
            train_hv = nn_planner.depth_encoder(train_batch_depths)
            train_recons = nn_planner.depth_decoder(train_hv)

            train_ae_loss = F.mse_loss(train_recons, train_batch_depths) + 0.5 * train_hv.pow(2).sum(-1).mean()
            train_epoch_ae_loss.append(train_ae_loss.item())

            optimizer.zero_grad()
            train_ae_loss.backward()
            optimizer.step()

        if args.log:
            sw.add_scalar("train/ae_loss", np.mean(train_epoch_ae_loss), global_step=i)
            sw.add_image("train/original", train_batch_depths.detach().cpu().numpy()[0, 0], global_step=i)
            sw.add_image("train/recon", train_recons.detach().cpu().numpy()[0, 0], global_step=i)

        # Validation
        nn_planner.eval()
        with torch.no_grad():
            for batch in replay_buffer.sample(args.batch_size, "val"):
                val_batch_depths, _, _ = batch

                # autoencoder forward pass
                val_hv = nn_planner.depth_encoder(val_batch_depths)
                val_recons = nn_planner.depth_decoder(val_hv)

                val_ae_loss = F.mse_loss(val_recons, val_batch_depths) + 0.5 * val_hv.pow(2).sum(-1).mean()
                val_epoch_ae_loss.append(val_ae_loss.item())

            if args.log:
                sw.add_scalar("val/ae_loss", np.mean(val_epoch_ae_loss), global_step=i)
                sw.add_image("val/original", val_batch_depths.detach().cpu().numpy()[0, 0], global_step=i)
                sw.add_image("val/recon", val_recons.detach().cpu().numpy()[0, 0], global_step=i)

        if args.log:
            sw.flush()

    # train neural policy
    # tqdm.write(f"Start training neural policy for {args.num_epochs} epochs")
    # for i in tqdm(range(args.num_epochs)):
    #     # Train
    #     nn_planner.train()
    #     for batch in replay_buffer.sample(args.batch_size, "train"):
    #         train_batch_depths, train_batch_states, train_batch_actions = batch

    #         h0, c0 = get_init_hidden_state(args.batch_size, args.hidden_dim, device)
    #         actions, (h, c) = nn_planner(train_batch_depths, train_batch_states, h0, c0, detach=True)

    #         # policy loss
    #         train_action_mse_loss = F.mse_loss(actions, train_batch_actions)
    #         train_epoch_action_mse_loss.append(train_action_mse_loss.item())

    #         optimizer.zero_grad()
    #         train_action_mse_loss.backward()
    #         optimizer.step()

    #         # ae loss
    #         train_hv = nn_planner.depth_encoder(train_batch_depths)
    #         train_recons = nn_planner.depth_decoder(train_hv)

    #         train_ae_loss = F.mse_loss(train_recons, train_batch_depths) + 0.5 * train_hv.pow(2).sum(-1).mean()
    #         train_epoch_ae_loss.append(train_ae_loss.item())

    #         optimizer.zero_grad()
    #         train_ae_loss.backward()
    #         optimizer.step()

    #     if args.log:
    #         sw.add_scalar("train/policy_loss", np.mean(train_epoch_action_mse_loss), global_step=i)
    #         sw.add_scalar("train/ae_loss", np.mean(train_epoch_ae_loss), global_step=args.num_pretrained_epochs+i)
    #         sw.add_image("train/original", train_batch_depths.detach().cpu().numpy()[0, 0], global_step=args.num_pretrained_epochs+i)
    #         sw.add_image("train/recon", train_recons.detach().cpu().numpy()[0, 0], global_step=args.num_pretrained_epochs+i)

    #     # Validation
    #     nn_planner.eval()
    #     with torch.no_grad():
    #         for batch in replay_buffer.sample(args.batch_size, "val"):
    #             val_batch_depths, val_batch_states, val_batch_actions = batch

    #             h0, c0 = get_init_hidden_state(args.batch_size, args.hidden_dim, device)
    #             actions, (h, c) = nn_planner(val_batch_depths, val_batch_states, h0, c0, detach=True)

    #             # policy loss
    #             val_action_mse_loss = F.mse_loss(actions, val_batch_actions)
    #             val_epoch_action_mse_loss.append(val_action_mse_loss.item())

    #             # ae loss
    #             val_hv = nn_planner.depth_encoder(val_batch_depths)
    #             val_recons = nn_planner.depth_decoder(val_hv)

    #             val_ae_loss = F.mse_loss(val_recons, val_batch_depths) + 0.5 * val_hv.pow(2).sum(-1).mean()
    #             val_epoch_ae_loss.append(val_ae_loss.item())

    #         if args.log:
    #             sw.add_scalar("val/policy_loss", np.mean(val_epoch_action_mse_loss), global_step=i)
    #             sw.add_scalar("val/ae_loss", np.mean(val_epoch_ae_loss), global_step=args.num_pretrained_epochs+i)
    #             sw.add_image("val/original", val_batch_depths.detach().cpu().numpy()[0, 0], global_step=args.num_pretrained_epochs+i)
    #             sw.add_image("val/recon", val_recons.detach().cpu().numpy()[0, 0], global_step=args.num_pretrained_epochs+i)

    #     # save model
    #     if args.save:
    #         if (i + 1) % args.save_freq == 0:
    #             torch.save(nn_planner.state_dict(), os.path.join(work_dir, f"model_{i}.pt"))

    #             curr_models = sorted(filter(lambda x: x.endswith("pt"), os.listdir(work_dir)), key=lambda x: re.search(r"\d+\.pt", x).group(0))
    #             if len(curr_models) > args.num_saved_models:
    #                 os.remove(os.path.join(work_dir, curr_models[0]))

    #     if args.log:
    #         sw.flush()


if __name__ == "__main__":
    args = parse_args()
    main(args)