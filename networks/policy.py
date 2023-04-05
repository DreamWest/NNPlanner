import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv1x1BNReLU(in_channels, out_channels, groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv1x1BN(in_channels, out_channels, groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    

class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = out_channels - in_channels if self.stride>1 else out_channels
        mid_channels = out_channels // 4

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels, groups),
            ChannelShuffle(groups),
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups),
            Conv1x1BN(mid_channels, out_channels, groups)
        )
        if self.stride>1:
            self.shortcut = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # TODO: check which is better? avg pooling or max pooling?

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat([self.shortcut(x), out], dim=1) if self.stride > 1 else (out + x)
        return self.relu(out)


class DepthEncoder(nn.Module):
    def __init__(self, groups=2, feature_dim=128):
        super().__init__()
        layers = [2, 4, 2]
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage2 = self._make_layer(32, 64, groups, layers[0])
        self.stage3 = self._make_layer(64, 128, groups, layers[1])
        self.stage4 = self._make_layer(128, 256, groups, layers[2])
        self.fc = nn.Linear(256 * 7 * 7, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def _make_layer(self, in_channels, out_channels, groups, block_num):
        layers = [ShuffleNetUnits(in_channels, out_channels, stride=2, groups=groups)]
        for i in range(1, block_num):
            layers.append(ShuffleNetUnits(out_channels, out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
                assert m.weight.size(2) == m.weight.size(3)
                m.weight.data.fill_(0.0)
                m.bias.data.fill_(0.0)
                mid = m.weight.size(2) // 2
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

    def forward(self, depth, detach=False):
        bs = depth.size(0)
        is_seq = False
        if len(depth.shape) == 5:
            is_seq = True
            seq_len = depth.size(1)
            n_c = depth.size(2)
            h, w = depth.size(3), depth.size(4)
            depth = depth.view(-1, n_c, h, w)
        feature1 = self.stage1(depth)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        h = self.stage4(feature3)

        if detach:
            h = torch.detach(h)

        if is_seq:
            h = h.view(bs * seq_len, -1)
        else:
            h = h.view(bs, -1)
        h = self.fc(h)
        h = torch.tanh(self.ln(h))

        if is_seq:
            h = h.view(bs, seq_len, -1)

        return h
    

class DepthDecoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, 256 * 7 * 7)
        self.relu = nn.ReLU6(inplace=True)

        self.upsampling1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsampling2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsampling3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upsampling4 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, padding=2)

    def forward(self, h):
        bs = h.size(0)
        is_seq = False
        if len(h.shape) == 3:
            is_seq = True
            seq_len = h.size(1)
            h = h.view(-1, self.feature_dim)
        h = self.relu(self.fc(h))
        
        h = h.view(-1, 256, 7, 7)
        h = self.upsampling1(h)
        h = self.upsampling2(h)
        h = self.upsampling3(h)
        recon = self.upsampling4(h)

        if is_seq:
            recon = recon.view(bs, seq_len, recon.size(1), recon.size(2), recon.size(3))
        return recon
    

class StateEncoder(nn.Module):
    def __init__(self, state_dim, feature_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, feature_dim//2)
        self.fc2 = nn.Linear(feature_dim//2, feature_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.fc2(out)
        return out
    

class NNPlanner(nn.Module):
    def __init__(self, state_dim, action_dim, groups=2, feature_dim=128, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth_encoder = DepthEncoder(groups, feature_dim)
        self.depth_decoder = DepthDecoder(feature_dim)
        self.state_encoder = StateEncoder(state_dim)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True),
            nn.Linear(256, action_dim)
        )
        self.lstm = nn.LSTM(2*feature_dim, hidden_dim, batch_first=True)

    def plan(self, depth, state, h, c, detach=False):
        hv = self.depth_encoder(depth, detach=detach)
        hs = self.state_encoder(state)
        hvs = torch.cat([hv, hs], dim=-1)
        out, (new_h, new_c) = self.lstm(hvs, (h, c))
        action = self.policy_net(out)
        return action, (new_h, new_c)

    def encode_decode(self, depth):
        hv = self.depth_encoder(depth)
        recon = self.depth_decoder(hv)
        return hv, recon

    def get_init_hidden_state(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_dim).to(device), torch.zeros(1, batch_size, self.hidden_dim).to(device)


if __name__ == "__main__":
    import time
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = torch.randn(batch_size, 10, 1, 224, 224).to(device)
    state = torch.randn(batch_size, 10, 13).to(device)
    # depth_encoder = DepthEncoder(4).to(device)
    # depth_decoder = DepthDecoder().to(device)
    # state_encoder = StateEncoder(13).to(device)
    # t0 = time.time()
    # hv = depth_encoder(depth)
    # print("depth hidden feat shape: {}".format(hv.shape))
    # t1 = time.time()
    # print("depth encoding time taken: {:.4f} s".format(t1-t0))
    # recon = depth_decoder(hv)
    # t2 = time.time()
    # print("depth decoding time taken: {:.4f} s".format(t2-t1))
    # hs = state_encoder(state)
    # print("state hidden feat shape: {}".format(hs.shape))
    # t3 = time.time()
    # print("state encoding time taken: {:.4f} s".format(t3-t2))
    nn_planner = NNPlanner(13, 9).to(device)
    nn_planner.train()
    h0, c0 = nn_planner.get_init_hidden_state(batch_size, device)
    t0 = time.time()
    action, (h, c) = nn_planner.plan(depth, state, h0, c0)
    t1 = time.time()
    hv, recon = nn_planner.encode_decode(depth)
    print("[training] planning time: {:.2f} ms".format(1000*(t1-t0)))
    print("----------------------------------input-----------------------------------")
    print("depth shape: {}, state shape: {}".format(depth.shape, state.shape))
    print("----------------------------------output----------------------------------")
    print("action shape: {}, h shape: {}, c shape: {}".format(action.shape, h.shape, c.shape))
    print("recon depth shape: {}".format(recon.shape))

    depth = torch.randn(1, 224, 224).to(device)
    state = torch.randn(13).to(device)
    depth = depth.unsqueeze(0).unsqueeze(0)
    state = state.unsqueeze(0).unsqueeze(0)

    nn_planner.eval()
    h0, c0 = nn_planner.get_init_hidden_state(1, device)
    with torch.no_grad():
        t0 = time.time()
        action, (h, c) = nn_planner.plan(depth, state, h0, c0)
        t1 = time.time()

    print("[inference] planning time: {:.2f} ms".format(1000*(t1-t0)))
    print("----------------------------------input-----------------------------------")
    print("depth shape: {}, state shape: {}".format(depth.shape, state.shape))
    print("----------------------------------output----------------------------------")
    print("action shape: {}, h shape: {}, c shape: {}".format(action.shape, h.shape, c.shape))
