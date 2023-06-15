import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def init_params(net):
    for m in net.modules():
        if (isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_init_hidden_state(batch_size, hidden_dim, device):
    return torch.zeros(1, batch_size, hidden_dim).to(device), torch.zeros(1, batch_size, hidden_dim).to(device)


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        init_params(self)

    def forward(self, depth, detach=False):
        batch_size = depth.size(0)
        is_sequence = False
        if len(depth.shape) == 5:
            is_sequence = True
            seq_length = depth.size(1)
            depth = depth.view(-1, depth.size(2), depth.size(3), depth.size(4))
        h = self.encoder(depth)

        if detach:
            h = torch.detach(h)

        if is_sequence:
            h = h.view(batch_size, seq_length, h.size(1), h.size(2), h.size(3))

        return h
    

class DepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        init_params(self)

    def forward(self, h):
        batch_size = h.size(0)
        is_sequence = False
        if len(h.shape) == 5:
            is_sequence = True
            seq_length = h.size(1)
            h = h.view(-1, 512, 7, 7)
        recon = self.decoder(h)

        if is_sequence:
            recon = recon.view(batch_size, seq_length, recon.size(1), recon.size(2), recon.size(3))

        return recon
    

class VisFeatExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, h):
        is_sequence = False
        if len(h.shape) == 5:
            is_sequence = True
            batch_size = h.size(0)
            seq_length = h.size(1)
            h = h.view(-1, h.size(2), h.size(3), h.size(4))
        h = self.avg_pool(h).squeeze()
        out = torch.relu(self.fc(h))
        if is_sequence:
            out = out.view(batch_size, seq_length, out.size(-1))
        return out


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
    def __init__(self, state_dim, action_dim, visual_feature_dim=128, state_feature_dim=128, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth_encoder = DepthEncoder()
        self.depth_decoder = DepthDecoder()
        self.vis_feat_extractor = VisFeatExtractor(visual_feature_dim)
        self.state_feat_extractor = StateEncoder(state_dim, state_feature_dim)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.lstm = nn.LSTM(visual_feature_dim, hidden_dim, batch_first=True)

    def forward(self, depth, state, h, c, detach=False):
        hv = self.depth_encoder(depth, detach=detach)           
        vf = self.vis_feat_extractor(hv)                # vf: (B, L, F_v)
        # sf = self.state_feat_extractor(state)
        sf = state                                      # sf: (B, L, F_s)
        out, (new_h, new_c) = self.lstm(vf, (h, c))     # out: (B, L, h_d)
        h = torch.cat([out, sf], dim=-1)
        action = self.policy_net(h)
        return action, (new_h, new_c)

        
if __name__ == "__main__":
    batch_size = 16
    seq_length = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = torch.randn(batch_size, seq_length, 1, 224, 224).to(device)
    state = torch.randn(batch_size, seq_length, 14).to(device)

    nn_planner = NNPlanner(14, 9, visual_feature_dim=128, state_feature_dim=64).to(device)
    h0, c0 = get_init_hidden_state(batch_size, 128, device)
    action, (h, c) = nn_planner(depth, state, h0, c0)
    print(action.shape)