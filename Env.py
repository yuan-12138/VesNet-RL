#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import cv2
import math
import torchvision.transforms as transforms
from Vessel_3d import Vessel_3d_sim, Vessel_3d
from collections import deque
import imutils
import sys

class Env_multi_sim_img():
    def __init__(self,configs,num_channels=4):
        self.configs=configs
        self.num_envs=len(configs)
        self.vessels=[]
        for config in configs:
            self.vessels.append(Vessel_3d_sim(config,probe_width=313))
        
        self.reward_window=deque(maxlen=num_channels+1)
        self.area_window=deque(maxlen=num_channels+1)
        
        self.z_size=[0,2*math.pi]
        self.actions=((0,0,0), (50, 0, 0), (-50, 0, 0), (0, 50, 0), (0, -50, 0), (0, 0, math.pi/18), (0, 0, -math.pi/18))
        
        self.num_actions=len(self.actions)
        self.num_channels=num_channels
        self.actions_all=[]
        self.action_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.action_his.append(-1)
        self.pose_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.pose_his.append(0)
        
    def reward_func(self):
        self.vessel_area=len(np.where(self.image>0.9)[0])
        max_area=self.vessels[self.cur_env].r*256/self.vessels[self.cur_env].size_3d[2]*256*2
        reward_vessel=(self.vessel_area-self.vessels[self.cur_env].threshold)/(max_area-self.vessels[self.cur_env].threshold)
        reward_dis=1-abs(self.pos[1]-self.vessels[self.cur_env].c[0])/(self.vessels[self.cur_env].probe_width/2+self.vessels[self.cur_env].r)
        return 0.7*reward_vessel+0.3*reward_dis
    

    def step(self,action_int):
        action=self.actions[action_int]
        
        new_pos=np.array([int(self.pos[0]+action[0]*np.cos(self.pos[2])-action[1]*np.sin(self.pos[2])),int(self.pos[1]+action[0]*np.sin(self.pos[2])+action[1]*np.cos(self.pos[2])),self.pos[2]+action[2]])
        
        if self.vessels[self.cur_env].check_mask(new_pos[0:2],new_pos[2]) and self.vessels[self.cur_env].vessel_existance(new_pos[0:2],new_pos[2]):
            self.pos=new_pos
            self.action_his.append(action_int)
            self.actions_all.append(action_int)
            self.pose_his.append(self.pos[2])
            reward_extra=-0.01
        else:
            self.pos=self.pos
            self.action_his.append(action_int)
            self.actions_all.append(-1)
            self.pose_his.append(self.pos[2])
            reward_extra=-0.1
            
        self.image,_,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        
        cur_reward=self.reward_func()
            
        state=self.image

        self.state.append(state)
        self.reward_window.append(cur_reward)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        done=False
        
        reward=self.reward_window[-1]-self.reward_window[-2]
        # reward=-self.reward_window[-1]+self.reward_window[-2]

        if cur_reward>0.9 and self.actions_all[-1]!=-1:
            if np.mean(self.reward_window)>0.95 and len(self.actions_all)>4:
                done=True
                reward=5
            else:
                reward=1

        
        
            
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]
        
        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000), reward+reward_extra, done
    
        
    def reset(self,randomVessel,randomStart=True):
        self.first_increase=True
        self.actions_all=[]
        
        for _ in range(self.reward_window.maxlen):
            self.reward_window.append(0)
        for _ in range(self.area_window.maxlen):
            self.area_window.append(0)
        for _ in range(self.action_his.maxlen):
            self.action_his.append(-1)
        for _ in range(self.pose_his.maxlen):
            self.pose_his.append(0)
        if randomVessel:
            self.cur_env=np.random.randint(self.num_envs)
        else:
            self.cur_env=0
        if randomStart:
            self.pos=self.vessels[self.cur_env].get_vertical_view(self.vessels[self.cur_env].size_3d[0]//2)
            while not (self.vessels[self.cur_env].check_mask(self.pos[0:2],self.pos[2])  and self.vessels[self.cur_env].vessel_existance(self.pos[0:2],self.pos[2])):
                    self.pos=self.vessels[self.cur_env].get_vertical_view_p(np.random.randint(self.vessels[self.cur_env].x_min+20,self.vessels[self.cur_env].x_max-20))        
        
        self.state=deque(maxlen=self.num_channels)
        for _ in range(self.state.maxlen):
            self.state.append(np.zeros([256,256]))
   
        self.image,self.poi,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        state=self.image
        
        self.state.append(state)
        cur_reward=self.reward_func()
        self.reward_window.append(cur_reward)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]

        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000)
    
class Env_multi_sim_img_test():
    def __init__(self,configs,num_channels=4):
        self.configs=configs
        self.num_envs=len(configs)
        self.vessels=[]
        for config in configs:
            self.vessels.append(Vessel_3d_sim(config,probe_width=313))
        
        self.reward_window=deque(maxlen=num_channels+1)
        self.area_window=deque(maxlen=num_channels+1)
        
        self.z_size=[0,2*math.pi]
        self.actions=((0,0,0), (50, 0, 0), (-50, 0, 0), (0, 50, 0), (0, -50, 0), (0, 0, math.pi/18), (0, 0, -math.pi/18))
        
        self.num_actions=len(self.actions)
        self.num_channels=num_channels
        self.actions_all=[]
        self.action_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.action_his.append(-1)
        self.pose_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.pose_his.append(0)
    
    def terminate_decision(self):
        if len(self.contours)<1:
            return False
        areas=[cv2.contourArea(c) for c in self.contours]
        max_area_index=np.argmax(areas)
        
        c=self.contours[max_area_index]
        area=areas[max_area_index]
        rect = cv2.minAreaRect(c)
        box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
        box = np.int0(box)
        box = np.clip(box, 0, 255)
        width=np.linalg.norm(box[0]-box[1])
        height=np.linalg.norm(box[1]-box[2])
        box_area=int(width*height)
        self.estimated_diameter.append(min(width,height))
        if areas[max_area_index]<5000:
            return False
        
        # terminate_coef=(box_area-area)/box_area
        # if terminate_coef<0.15 and max(width,height) >250 and min(width,height)>(np.mean(self.estimated_diameter)-10):
        #     return True
        # else:
        #     return False
        
        terminate_coef=(box_area-area)/box_area
        if max(width,height) >250 and min(width,height)>(np.mean(self.estimated_diameter)-10) and terminate_coef<0.1:
            return True
        else:
            return False
    

    def step(self,action_int):
        action=self.actions[action_int]
        
        new_pos=np.array([int(self.pos[0]+action[0]*np.cos(self.pos[2])-action[1]*np.sin(self.pos[2])),int(self.pos[1]+action[0]*np.sin(self.pos[2])+action[1]*np.cos(self.pos[2])),self.pos[2]+action[2]])
        
        if self.vessels[self.cur_env].check_mask(new_pos[0:2],new_pos[2]) and self.vessels[self.cur_env].vessel_existance(new_pos[0:2],new_pos[2]):
            self.pos=new_pos
            self.action_his.append(action_int)
            self.actions_all.append(action_int)
            self.pose_his.append(self.pos[2])
        else:
            self.pos=self.pos
            self.action_his.append(action_int)
            self.actions_all.append(-1)
            self.pose_his.append(self.pos[2])
            
        self.image,_,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        
        self.uint_img = np.array(self.image).astype('uint8')
        
        self.contours,_ = cv2.findContours(self.uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
        state=self.image

        self.state.append(state)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        done=False
        

        done=self.terminate_decision()

        
        
            
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]
        
        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000), done
    
        
    def reset(self,randomVessel,randomStart=True):
        self.first_increase=True
        self.actions_all=[]
        self.estimated_diameter=deque(maxlen=10)
        for _ in range(self.reward_window.maxlen):
            self.reward_window.append(0)
        for _ in range(self.area_window.maxlen):
            self.area_window.append(0)
        for _ in range(self.action_his.maxlen):
            self.action_his.append(-1)
        for _ in range(self.pose_his.maxlen):
            self.pose_his.append(0)
        if randomVessel:
            self.cur_env=np.random.randint(self.num_envs)
        else:
            self.cur_env=0
        if randomStart:
            self.pos=self.vessels[self.cur_env].get_vertical_view(self.vessels[self.cur_env].size_3d[0]//2)
            while not (self.vessels[self.cur_env].check_mask(self.pos[0:2],self.pos[2])  and self.vessels[self.cur_env].vessel_existance(self.pos[0:2],self.pos[2])):
                    self.pos=self.vessels[self.cur_env].get_vertical_view_p(np.random.randint(self.vessels[self.cur_env].x_min+20,self.vessels[self.cur_env].x_max-20))        
        
        self.state=deque(maxlen=self.num_channels)
        for _ in range(self.state.maxlen):
            self.state.append(np.zeros([256,256]))
   
        self.image,self.poi,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        state=self.image
        
        self.state.append(state)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]

        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000)

class Env_multi_re_img_a2c_test():
    def __init__(self,n1_img,num_channels=4,points_interval=100,reward_space=None):
        self.num_envs=len(n1_img)
        self.vessels=[]
        # for file in n1_img:
        #     self.vessels.append(Vessel_3d(file[0],probe_width=313))
        #     self.vessels[-1].get_vessel_centerline(points_interval,file[1])
        for file in n1_img:
            self.vessels.append(Vessel_3d(file[0],probe_width=313))
            self.vessels[-1].get_vessel_centerline(points_interval,file[1])
        
        self.reward_window=deque(maxlen=num_channels+1)
        self.area_window=deque(maxlen=num_channels+1)
        self.version=sys.version[0]
        self.z_size=[-np.pi,7*math.pi/6]
        self.actions=((0,0,0), (50, 0, 0), (-50, 0, 0), (0, 50, 0), (0, -50, 0), (0, 0, math.pi/18), (0, 0, -math.pi/18))
        self.num_actions=len(self.actions)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        
        if self.version=='2':
            self.transform_image = transforms.Compose([
                #transforms.Resize(resize_to),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])
        else:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5,0.5)
            ])
        
        self.num_channels=num_channels
        self.actions_all=[]
        self.action_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.action_his.append(-1)
        self.pose_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.pose_his.append(0)
        
        self.estimated_diameter=deque(maxlen=10)
        
    def terminate_decision(self):
        if len(self.contours)<1:
            return False
        areas=[cv2.contourArea(c) for c in self.contours]
        max_area_index=np.argmax(areas)
        
        c=self.contours[max_area_index]
        area=areas[max_area_index]
        rect = cv2.minAreaRect(c)
        box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
        box = np.int0(box)
        box = np.clip(box, 0, 255)
        width=np.linalg.norm(box[0]-box[1])
        height=np.linalg.norm(box[1]-box[2])
        box_area=int(width*height)
        self.estimated_diameter.append(min(width,height))
        if areas[max_area_index]<5000:
            return False
        
        terminate_coef=(box_area-area)/box_area
        if terminate_coef<0.15 and max(width,height) >250 and min(width,height)>(np.mean(self.estimated_diameter)-10):
            return True
        else:
            return False
    
    def step(self,action_int):
        action=self.actions[action_int]
        
        new_pos=np.array([int(self.pos[0]+action[0]*np.cos(self.pos[2])-action[1]*np.sin(self.pos[2])),int(self.pos[1]+action[0]*np.sin(self.pos[2])+action[1]*np.cos(self.pos[2])),self.pos[2]+action[2]])
        
        if self.vessels[self.cur_env].check_mask(new_pos[0:2],new_pos[2]) and self.vessels[self.cur_env].vessel_existance(new_pos[0:2],new_pos[2]):
            self.pos=new_pos
            self.action_his.append(action_int)
            self.actions_all.append(action_int)
            self.pose_his.append(self.pos[2])

        else:
            self.pos=self.pos
            self.action_his.append(action_int)
            self.actions_all.append(-1)
            self.pose_his.append(self.pos[2])

            
        self.image,_,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        
        x = self.transform_image(self.image)
        x=x.view(-1, 1, 256, 256).float().to(self.device)
        pred_tensor=self.vessels[self.cur_env].unet_best(x)
        pred=pred_tensor.view(256, 256).cpu().detach().numpy()
        _,self.pred_th=cv2.threshold(pred,0.5,1.0,0)
        
        state=self.pred_th
        self.state.append(state)
        done=False
        
        self.vessel_area=len(np.where(self.pred_th>0.9)[0])
        
        self.uint_img = np.array(self.pred_th*255).astype('uint8')
        if self.version=='2':
            _,self.contours,_ = cv2.findContours(self.uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        else:
            self.contours,_ = cv2.findContours(self.uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        too_many_contours=len(self.contours)>5
        
        if too_many_contours:
            self.area_window.append(0)
        else:
            self.area_window.append(self.vessel_area)
        
        done=self.terminate_decision()
            
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]
        
        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000.0), done
    
        
    def reset(self,randomVessel,randomStart=True):
        self.first_increase=True
        self.actions_all=[]
        self.estimated_diameter=deque(maxlen=10)
        for _ in range(self.reward_window.maxlen):
            self.reward_window.append(0)
        for _ in range(self.area_window.maxlen):
            self.area_window.append(0)
        for _ in range(self.action_his.maxlen):
            self.action_his.append(-1)
        for _ in range(self.pose_his.maxlen):
            self.pose_his.append(0)
        if randomVessel:
            self.cur_env=np.random.randint(self.num_envs)
        else:
            self.cur_env=0
        if randomStart:
            self.pos=self.vessels[self.cur_env].get_vertical_view(np.random.randint(self.vessels[self.cur_env].x_min+200,self.vessels[self.cur_env].x_max-200))
            while not (self.vessels[self.cur_env].check_mask(self.pos[0:2],self.pos[2]) and self.vessels[self.cur_env].vessel_existance(self.pos[0:2],self.pos[2])):
                    self.pos=self.vessels[self.cur_env].get_vertical_view(np.random.randint(self.vessels[self.cur_env].x_min+200,self.vessels[self.cur_env].x_max-200))
        
        self.state=deque(maxlen=self.num_channels)
        for _ in range(self.state.maxlen):
            self.state.append(np.zeros([256,256]))
   
        self.image,self.poi,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        
        x = self.transform_image(self.image)
        x=x.view(-1, 1, 256, 256).float().to(self.device)
        pred_tensor=self.vessels[self.cur_env].unet_best(x)
        pred=pred_tensor.view(256, 256).cpu().detach().numpy()
        _,self.pred_th=cv2.threshold(pred,0.5,1.0,0)
        
        state=self.pred_th
        
        self.state.append(state)
        self.vessel_area=len(np.where(self.pred_th>0.9)[0])
        self.area_window.append(self.vessel_area)
        self.uint_img = np.array(self.pred_th*255).astype('uint8')
        if self.version=='2':
            _,self.contours,_ = cv2.findContours(self.uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        else:
            self.contours,_ = cv2.findContours(self.uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        done=self.terminate_decision()
        
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]
        
        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000.0)
    
    