#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:41:25 2021

@author: robotics
"""
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from Env import Env_multi_sim_img
from model import VesNet_RL
from collections import deque
import matplotlib.pyplot as plt

LR = 5e-4
max_step=500
n_episodes=3000
gamma=0.99
gae_lambda=1.0
entropy_coef=0.01
value_loss_coef=0.5
max_grad_norm=50
save_every=50
update_every=20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_configs_rand(num):
    configs=[]
    r_min=30
    r_max=75
    for i in range(num):
        offset=np.random.rand()*np.pi/2
        size_3d=[750,700,450]
        r=np.random.randint(r_min+(r_max-r_min)*i/num,r_min+(r_max-r_min)*(i+1)/num)
        c_x=350
        c_y=np.random.randint(50+r,225)
        c=[c_x,c_y]
        config=(c,r,size_3d,offset)
        configs.append(config)
    return configs

configs=create_configs_rand(10)

env=Env_multi_sim_img(configs=configs, num_channels=4)

model = VesNet_RL(env.num_channels, 5, env.num_actions).to(device)


optimizer = optim.Adam(model.parameters(), lr=LR)

scores_window = deque(maxlen=50)
rewards_window = deque(maxlen=50)
rewards_his=[]
smoothend_rewards=[]

a_file = open("VesNet_RL_ckpt/configs.txt", "w")
for row in configs:
    a_file.write(str(row)+'\r')
a_file.close()


plt.clf()
plt.ylabel('Score')
plt.xlabel('Episode #')
done_his=deque(maxlen=100)
reward_max=-sys.float_info.max
best_success_rate=0
i_episode=0
done=True
t=0

state = env.reset(randomVessel=False)
reward_sum=0
finish=False

while i_episode<n_episodes:
    if i_episode==1500:
        for g in optimizer.param_groups:
            g['lr'] = 1e-4
    elif i_episode==500:
        for g in optimizer.param_groups:
            g['lr'] = 3e-4
    elif i_episode==0:
        for g in optimizer.param_groups:
            g['lr'] = 5e-4
    
    if done:
        cx = torch.zeros(1, 256).float().to(device)
        hx = torch.zeros(1, 256).float().to(device)
    else:
        cx = cx.detach()
        hx = hx.detach()

    values = []
    log_probs = []
    rewards = []
    entropies = []
    
    for _ in range(0,update_every):
        t+=1
        value, logit, (hx, cx) = model((state,(hx, cx)))
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        entropies.append(entropy)
        
        action = prob.multinomial(num_samples=1).detach()
        log_prob = log_prob.gather(1, action)
        state, reward, finish_ = env.step(int(action.cpu().detach().numpy()))
        finish=finish or finish_
        done = t >= max_step
        
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(reward)
        
        reward_sum+=reward

        if done:
            break
        
    
    R = torch.zeros(1, 1).to(device)
    if not done:
        value, _, _ = model((state, (hx, cx)))
        R = value.detach()

    values.append(R)
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1).to(device)
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = rewards[i] + gamma * values[i + 1] - values[i]
        gae = gae * gamma * gae_lambda + delta_t

        policy_loss = policy_loss - log_probs[i] * gae.detach() - entropy_coef * entropies[i]

    optimizer.zero_grad()

    (policy_loss + value_loss_coef * value_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    
    
    if done:
        done_his.append(finish)
        scores_window.append(reward_sum)
        rewards_window.append(reward_sum)
        rewards_his.append(reward_sum)
        smoothend_rewards.append(np.mean(scores_window))
        
        if np.mean(rewards_window)>reward_max:
            reward_max=np.mean(rewards_window)
            torch.save(model.state_dict(), 'VesNet_RL_ckpt/checkpoint.pth')
            a_file = open("VesNet_RL_ckpt/best_ckpt.txt", "w")
            a_file.write(str([i_episode,reward_max])+'\r')
            a_file.close()
        
        if i_episode%save_every==0:
            torch.save(model.state_dict(), 'VesNet_RL_ckpt/checkpoint_latest.pth')
        success_rate=len(np.where(np.array(done_his)==1)[0])
        
        if success_rate>best_success_rate:
            best_success_rate=success_rate
            if i_episode>100:
                torch.save(model.state_dict(), 'VesNet_RL_ckpt/checkpoint_best_sr.pth')
                a_file = open("VesNet_RL_ckpt/best_ckpt_sr.txt", "w")
                a_file.write(str([i_episode,best_success_rate])+'\r')
                a_file.close()
        print('\r Episode %d Average Score: %.2f Score: %.2f Steps: %d Done: %s Best success rate: %d %% Success rate: %d %%\r' % 
                             (i_episode,np.mean(scores_window),reward_sum,t,finish,best_success_rate,success_rate), end='')
        
        
        if len(rewards_his)>2:
            plt.plot([len(rewards_his)-1,len(rewards_his)], [rewards_his[-2],rewards_his[-1]],'b', alpha=0.3)
            plt.plot([len(smoothend_rewards)-1,len(smoothend_rewards)], [smoothend_rewards[-2],smoothend_rewards[-1]],'g')
            plt.pause(0.05)
        
        t = 0
        if i_episode>50:
            state = env.reset(randomVessel=True)
        else:
            state = env.reset(randomVessel=True)
        i_episode+=1
        reward_sum=0
        finish=False
        

torch.save(model.state_dict(), 'VesNet_RL_ckpt/checkpoint_latest.pth')
plt.show()