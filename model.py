import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class VesNet_RL(torch.nn.Module):
    def __init__(self, num_channels, z_dim, num_actions):
        super(VesNet_RL, self).__init__()
        
        self.Conv=nn.Sequential(
			nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
			nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
			nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
			nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
			nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, z_dim*num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(z_dim*num_channels),
            nn.ReLU(),
            nn.AvgPool2d(4),
            
            Flatten()
		)

        self.lstm = nn.LSTMCell(num_channels*(z_dim+2), 256)

        self.critic_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )

        self.actor_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
            )

        self.apply(weights_init)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)


    def forward(self, inputs):
        (image,actions,area_changes), (hx, cx) = inputs

        image=torch.from_numpy(image).float().to('cuda')
        z=self.Conv(image.unsqueeze(0)).flatten()
        actions=torch.from_numpy(actions).float().to('cuda')
        area_changes=torch.from_numpy(area_changes).float().to('cuda')

        inputs = torch.cat((z,actions,area_changes),dim=0).flatten()
        hx, cx = self.lstm(inputs.unsqueeze(0), (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    