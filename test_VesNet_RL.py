#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:38:59 2021

@author: robotics
"""
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from Env import Env_multi_sim_img_test
from model import VesNet_RL
from collections import deque
import matplotlib.pyplot as plt

n_episodes=100
max_step=50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_configs_rand(num):
    configs=[]
    r_min=45
    r_max=65
    for i in range(num):
        offset=np.random.rand()*np.pi/2
        size_3d=[750,700,450]
        r=np.random.randint(r_min+(r_max-r_min)*i/num,r_min+(r_max-r_min)*(i+1)/num)
        c_x=350
        c_y=np.random.randint(r,450-r)
        c=[c_x,c_y]
        config=(c,r,size_3d,offset)
        configs.append(config)
    return configs

configs=create_configs_rand(1)

env=Env_multi_sim_img_test(configs=configs, num_channels=4)

model = VesNet_RL(env.num_channels, 5, env.num_actions).to(device)

model.load_state_dict(torch.load('VesNet_RL_ckpt/trained_model/checkpoint.pth',map_location='cuda'))

episode_length = 0
done = True
actions_his=[]
rewards_his=[]
num_success=0
init_states=[]
pos_his=[]
success_num=np.zeros(10)
env_his=np.zeros(10)
for i_episode in range(1, n_episodes+1):
    state = env.reset(randomVessel=True)
    env_his[env.cur_env]+=1
    init_states.append(state)
    init_pos=env.pos
    reward_sum = 0
    
    done=True
    if done:
        cx = torch.zeros(1, 256).float().to(device)
        hx = torch.zeros(1, 256).float().to(device)
    else:
        cx = cx.detach()
        hx = hx.detach()
    t=0
    actions=[]
    positions=[]
    probs_max=[]
    for step in range(max_step):
        t+=1
        with torch.no_grad():
            # value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            value, logit, (hx, cx) = model((state, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        # action = prob.max(1, keepdim=True)[1].cpu().detach().numpy()
        action = prob.multinomial(num_samples=1).cpu().detach().numpy()
        probs_max.append(prob.cpu().detach().numpy())
    
        state, done = env.step(int(action))
    
        actions.append(int(action))
        positions.append(env.pos)
        if done:
            num_success+=1
            success_num[env.cur_env]+=1
            break
    actions_his.append(actions)
    print('\r Episode %d Steps: %d Done: %s\r' % 
                          (i_episode,t,done), end='')

    