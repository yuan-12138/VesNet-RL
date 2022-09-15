#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:13:06 2021

@author: eadu
"""

import numpy as np
# import matplotlib.pyplot as plt
import torch
import itertools
import math
import cv2
from scipy.spatial.transform import Rotation
import imutils
import torchvision.transforms as transforms
import nibabel as nib
from UNet import UNet
import sys
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def create_vessel(c,r,size_3d):
    img_2d=np.random.rand(size_3d[1],size_3d[2])*0.01
    for i in range(size_3d[1]):
        for j in range(size_3d[2]):
            if (i-c[0])**2+(j-c[1])**2<=r**2:
                img_2d[i,j]=1
    return np.repeat(img_2d[np.newaxis, :, :], size_3d[0], axis=0)

class Vessel_3d_sim:

    def __init__(self, config, probe_width=313):
        self.img = create_vessel(config[0],config[1],config[2])
        self.c=config[0]
        self.r=config[1]
        self.size_3d=config[2]
        self.probe_width=probe_width
        
        self.mask=np.ones(self.img.shape[0:2])
        for z in np.arange(40,self.img.shape[2]-40):
            image=self.img[:,:,z]
            _,thresh = cv2.threshold(image,0,1,cv2.THRESH_BINARY)
            thresh=cv2.blur(thresh,(25,25))
            _,thresh = cv2.threshold(thresh,0.15,1,cv2.THRESH_BINARY)
            self.mask=self.mask*thresh
        self.mask[0:10]=np.zeros([10,self.img.shape[1]])
        self.mask[-10:-1]=np.zeros([9,self.img.shape[1]])
        self.mask[-1]=np.zeros([1,self.img.shape[1]])
        self.mask[:,0:4]=np.zeros([self.img.shape[0],4])
        self.mask[:,-4:-1]=np.zeros([self.img.shape[0],3])
        self.mask[:,-1]=np.zeros([1,self.img.shape[0]])
        
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ])
        
        self.threshold=100
        self.x_min=160
        self.x_max=self.size_3d[0]-160
    
    def voxel_to_base(self,voxel_pose):
        # voxel_pose [x,y,z,rz]
        
        voxel_pos=np.array([voxel_pose[0],voxel_pose[1],voxel_pose[2],1])
        x=np.dot(self.n1_header['srow_x'],np.transpose(voxel_pos))
        y=np.dot(self.n1_header['srow_y'],np.transpose(voxel_pos))
        z=np.dot(self.n1_header['srow_z'],np.transpose(voxel_pos))
        base_pos=np.array([-x,-y,z])
        
        if voxel_pose[3]>math.pi/2:
            voxel_pose[3]-=math.pi
        r=Rotation.from_euler('z', -voxel_pose[3], degrees=False)
        r_robot=Rotation.from_quat([0,1,0,0])
        r_final=r_robot*r
        base_quat=r_final.as_quat()
        
        best_pose=np.concatenate((base_pos[0:3]/1000, base_quat), axis=None)
        return best_pose
    
    def check_mask(self,p,theta):
        result=False
        try:
            temp1=self.mask[int(p[0]+(self.probe_width)*math.cos(theta)*0.5),int(p[1]+(self.probe_width)*math.sin(theta)*0.5)]==0
            temp2=self.mask[int(p[0]-(self.probe_width)*math.cos(theta)*0.5),int(p[1]-(self.probe_width)*math.sin(theta)*0.5)]==0
            temp3=int(p[0]+(self.probe_width)*math.cos(theta)*0.5)<0
            temp4=int(p[1]+(self.probe_width)*math.sin(theta)*0.5)<0
            temp5=int(p[0]-(self.probe_width)*math.cos(theta)*0.5)<0
            temp6=int(p[1]-(self.probe_width)*math.sin(theta)*0.5)<0
            result=result+temp1+temp2+temp3+temp4+temp5+temp6
            for _ in range(10):
                if result:
                    break
                i=np.random.rand()*0.5
                temp1=self.mask[int(p[0]+(self.probe_width)*math.cos(theta)*i),int(p[1]+(self.probe_width)*math.sin(theta)*i)]==0
                temp2=self.mask[int(p[0]-(self.probe_width)*math.cos(theta)*i),int(p[1]-(self.probe_width)*math.sin(theta)*i)]==0
                result=result+temp1+temp2
            return not result
        except:
            return False
        
        
    def get_searching_points(self,points_interval,theta):
        # searching_area [x_size,y_size]
        # probe_width width of the us probe in pixel, 313 by default (37.5mm)
        # points_interval is the distance between points, 10 (pixel) by default (1.2mm)
        # theta the angle from x axis to the line, which represents the probe [0,pi]
        valid_area=self.mask.shape
        num_points=(np.array(valid_area)-np.array([abs(self.probe_width*math.cos(theta)),abs(self.probe_width*math.sin(theta))]))//points_interval
        num_points=num_points.clip(min=1)
        p_x=np.arange(num_points[0])-((num_points[0]-1)/2)
        p_y=np.arange(num_points[1])-((num_points[1]-1)/2)
        p_x=list(p_x*points_interval+valid_area[0]//2)
        p_y=list(p_y*points_interval+valid_area[1]//2)
        searching_points=itertools.product([int(i) for i in p_x],[int(i) for i in p_y])
        searching_points=list(searching_points)
        delete_list=[]
        for i in range(len(searching_points)):
            p=searching_points[i]
            if not self.check_mask(p,theta):
                delete_list.append(i)
        delete_list=sorted(delete_list, reverse=True)
        for j in delete_list:
            del searching_points[j]
        return [list(e)+[theta] for e in searching_points]
    
    
    def merge_image(self,poi):
        image=[]
        for p in poi:
            image.append(list(self.img[p[0],p[1],:]))
        image=np.array(image)
        try:
            image=image[~np.all(image == 0, axis=1)]
            image=image[:,~np.all(image == 0, axis=0)]
            image=cv2.resize(image, (256,256),interpolation=cv2.INTER_NEAREST)
            image=imutils.rotate(image,90)
            return image,True
        except:
            return image,False
        
        
        
    def get_slicer(self, center_point,theta):
        # image_3d 3d array
        # center_point the center point of the us probe (np array)
        # theta the angle from x axis to the line, which represents the probe [0,pi]
        # probe_width width of the us probe in pixel, 313 by default (37.5mm)
        poi_tmp=[]
        poi=[]
        rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        for i in np.arange(-self.probe_width//2,self.probe_width//2):
            p=np.dot(rot,np.transpose(np.array([i,0])))
            p=p+np.array(center_point)
            poi_tmp.append([int(p[0]),int(p[1])])
        for i in poi_tmp:
            if i not in poi:
                poi.append(i)
        image,success=self.merge_image(poi)
        return image, poi, success
    
    
    def get_image_centroid(self,image):
        # image should be float format [0,1]
        uint_img = np.array(image*255).astype('uint8')
    
        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(uint_img,127,255,0)
    
        # find contours in the binary image
        contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     print(hierarchy)
        
        centroids=[]
        for c in contours:
            area=cv2.contourArea(c)
            if area<1000:
                continue
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                continue
            centroids.append([cX,cY])
    #         cv2.circle(uint_img, (cX, cY), 1, (0, 0, 0), -1)
    
    #     uint_img=uint_img/255
    #     plt.imshow(uint_img,cmap="gray")
        return np.array(centroids)
    
    def find_pixel_pose(self, pixel_pose_img, probe_pose):
        # find the position of the vessel center in x-y plane of 3d image pixel wise
        # position of the center of vessel in simulated US images [x,y]
        # probe_pose [x,y,theta] in pixel
        theta=probe_pose[2]
        x_img=pixel_pose_img[0]
        result=[probe_pose[0]+(x_img-self.probe_width/2)*math.cos(theta),probe_pose[1]+(x_img-self.probe_width/2)*math.sin(theta)]
        
        return [int(e) for e in result]
    
    def get_parallel_view(self, x):
        return self.get_slicer([x,self.c[0]],0)
    
    def get_vertical_view(self, x):
        y=self.c[0]+(-1)**np.random.randint(2)*np.random.randint(0,50)
        
        return [x,y,np.pi/2]
    
    def get_vertical_view_p(self,x):
        return [x,self.c[0],(np.random.rand()*2+17)*np.pi/36]
    
    def vessel_existance(self,pos,theta):
        img,_,_=self.get_slicer(pos,theta)
        area=len(np.where(img>0.9)[0])
        if area<self.threshold:
            return False
        else:
            return True





class Vessel_3d:

    def __init__(self, ni1_file, unet_file='./UNet_ckpt/unet_usseg_phantom.pth', probe_width=313):
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.n1_img = nib.load(ni1_file)
        self.img = self.n1_img.get_fdata()
        self.probe_width=probe_width
        self.n1_header=self.n1_img.header
        self.unet_best = UNet(init_features=64).to(self.device)
        self.unet_best.load_state_dict(torch.load(unet_file))
        self.unet_best = self.unet_best.eval()
        # self.unet_best=unet_best
        self.version=sys.version[0]
        self.mask=np.ones(self.img.shape[0:2])
        for z in np.arange(40,self.img.shape[2]-40):
            image=self.img[:,:,z]
            _,thresh = cv2.threshold(image,0,1,cv2.THRESH_BINARY)
            thresh=cv2.blur(thresh,(25,25))
            _,thresh = cv2.threshold(thresh,0.15,1,cv2.THRESH_BINARY)
            self.mask=self.mask*thresh
        self.mask[0:10]=np.zeros([10,self.img.shape[1]])
        self.mask[-10:-1]=np.zeros([9,self.img.shape[1]])
        self.mask[-1]=np.zeros([1,self.img.shape[1]])
        self.mask[:,0:4]=np.zeros([self.img.shape[0],4])
        self.mask[:,-4:-1]=np.zeros([self.img.shape[0],3])
        self.mask[:,-1]=np.zeros([1,self.img.shape[0]])
        if self.version=='2':
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
            ])
        else:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5,0.5)
            ])
        
        self.threshold=1500
        
        self.max_area=0
    
    def find_goal_pos(self,points_interval):
        x_min=200
        x_max=self.img.shape[0]-200
        y_min=200
        y_max=self.img.shape[1]-200
        theta=np.arctan(self.coef[1])
        if theta<0:
            theta+=math.pi
        for x_p in np.arange(x_min,x_max,points_interval):
            for y_p in np.arange(y_min,y_max,points_interval):
                img,_,success=self.get_slicer([x_p,y_p],theta)
                if not success:
                    continue
                x = self.transform_image(img)
                x=x.view(-1, 1, 256, 256).float().to(self.device)
                pred_tensor=self.unet_best(x)
                pred=pred_tensor.view(256, 256).cpu().detach().numpy()
                _,pred_th=cv2.threshold(pred,0.5,1.0,0)
                uint_img = np.array(pred_th*255).astype('uint8')
                if self.version=='2':
                    _,contours,_ = cv2.findContours(uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                else:
                    contours,_ = cv2.findContours(uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)>3:
                    continue
                area=len(np.where(pred_th>0.9)[0])
                # area_records.append(area)
                # pos_records.append([x_in,y,theta])
                if area>self.max_area:
                    self.max_area=area
                    self.goal_pos=[x_p,y_p,theta]
    
    def voxel_to_base(self,voxel_pose):
        # voxel_pose [x,y,z,rz]
        
        voxel_pos=np.array([voxel_pose[0],voxel_pose[1],voxel_pose[2],1])
        x=np.dot(self.n1_header['srow_x'],np.transpose(voxel_pos))
        y=np.dot(self.n1_header['srow_y'],np.transpose(voxel_pos))
        z=np.dot(self.n1_header['srow_z'],np.transpose(voxel_pos))
        base_pos=np.array([-x,-y,z])
        
        if voxel_pose[3]>math.pi/2:
            voxel_pose[3]-=math.pi
        r=Rotation.from_euler('z', -voxel_pose[3], degrees=False)
        r_robot=Rotation.from_quat([0,1,0,0])
        r_final=r_robot*r
        base_quat=r_final.as_quat()
        
        best_pose=np.concatenate((base_pos[0:3]/1000, base_quat), axis=None)
        return best_pose
    
    def check_mask(self,p,theta):
        result=False
        try:
            temp1=self.mask[int(p[0]+(self.probe_width)*math.cos(theta)*0.5),int(p[1]+(self.probe_width)*math.sin(theta)*0.5)]==0
            temp2=self.mask[int(p[0]-(self.probe_width)*math.cos(theta)*0.5),int(p[1]-(self.probe_width)*math.sin(theta)*0.5)]==0
            temp3=int(p[0]+(self.probe_width)*math.cos(theta)*0.5)<0
            temp4=int(p[1]+(self.probe_width)*math.sin(theta)*0.5)<0
            temp5=int(p[0]-(self.probe_width)*math.cos(theta)*0.5)<0
            temp6=int(p[1]-(self.probe_width)*math.sin(theta)*0.5)<0
            result=result+temp1+temp2+temp3+temp4+temp5+temp6
            for _ in range(10):
                if result:
                    break
                i=np.random.rand()*0.5
                temp1=self.mask[int(p[0]+(self.probe_width)*math.cos(theta)*i),int(p[1]+(self.probe_width)*math.sin(theta)*i)]==0
                temp2=self.mask[int(p[0]-(self.probe_width)*math.cos(theta)*i),int(p[1]-(self.probe_width)*math.sin(theta)*i)]==0
                result=result+temp1+temp2
            return not result
        except:
            return False
        
        
    def get_searching_points(self,points_interval,theta):
        # searching_area [x_size,y_size]
        # probe_width width of the us probe in pixel, 313 by default (37.5mm)
        # points_interval is the distance between points, 10 (pixel) by default (1.2mm)
        # theta the angle from x axis to the line, which represents the probe [0,pi]
        valid_area=self.mask.shape
        num_points=(np.array(valid_area)-np.array([abs(self.probe_width*math.cos(theta)),abs(self.probe_width*math.sin(theta))]))//points_interval
        num_points=num_points.clip(min=1)
        p_x=np.arange(num_points[0])-((num_points[0]-1)/2)
        p_y=np.arange(num_points[1])-((num_points[1]-1)/2)
        p_x=list(p_x*points_interval+valid_area[0]//2)
        p_y=list(p_y*points_interval+valid_area[1]//2)
        searching_points=itertools.product([int(i) for i in p_x],[int(i) for i in p_y])
        searching_points=list(searching_points)
        delete_list=[]
        for i in range(len(searching_points)):
            p=searching_points[i]
            if not self.check_mask(p,theta):
                delete_list.append(i)
        delete_list=sorted(delete_list, reverse=True)
        for j in delete_list:
            del searching_points[j]
        return [list(e)+[theta] for e in searching_points]
    
    
    def merge_image(self,poi):
        image=[]
        for p in poi:
            image.append(list(self.img[p[0],p[1],:]))
        image=np.array(image)
        try:
            image=image[~np.all(image == 0, axis=1)]
            image=image[:,~np.all(image == 0, axis=0)]
            image=cv2.resize(image, (256,256),interpolation=cv2.INTER_LANCZOS4)
            image=imutils.rotate(image,90)
            image=image.astype(int)/255.0
            return image,True
        except:
            return image,False
        
        
        
    def get_slicer(self, center_point,theta):
        # image_3d 3d array
        # center_point the center point of the us probe (np array)
        # theta the angle from x axis to the line, which represents the probe [0,pi]
        # probe_width width of the us probe in pixel, 313 by default (37.5mm)
        poi_tmp=[]
        poi=[]
        rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        for i in np.arange(-self.probe_width//2,self.probe_width//2):
            p=np.dot(rot,np.transpose(np.array([i,0])))
            p=p+np.array(center_point)
            poi_tmp.append([int(p[0]),int(p[1])])
        for i in poi_tmp:
            if i not in poi:
                poi.append(i)
        image,success=self.merge_image(poi)
        return image, poi, success
    
    
    def get_image_centroid(self,image):
        # image should be float format [0,1]
        uint_img = np.array(image*255).astype('uint8')
    
        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(uint_img,127,255,0)
    
        # find contours in the binary image
        if self.version=='2':
            _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        centroids=[]
        for c in contours:
            area=cv2.contourArea(c)
            if area<1000:
                continue
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                continue
            centroids.append([cX,cY])
        return np.array(centroids), len(contours)
    
    def find_pixel_pose(self, pixel_pose_img, probe_pose):
        # find the position of the vessel center in x-y plane of 3d image pixel wise
        # position of the center of vessel in simulated US images [x,y]
        # probe_pose [x,y,theta] in pixel
        theta=probe_pose[2]
        x_img=self.probe_width*pixel_pose_img[0]/256
        result=[probe_pose[0]+(x_img-self.probe_width/2)*math.cos(theta),probe_pose[1]+(x_img-self.probe_width/2)*math.sin(theta)]
        
        return [int(e) for e in result]
        
    
    def get_vessel_centerline(self,points_interval,theta):

        list_IDs=[]
        points=self.get_searching_points(points_interval,theta)
        list_IDs+=points
            
        self.center_line=np.zeros(self.img.shape[0:2])
        num_p=0
        for p in list_IDs:
            img,_,_=self.get_slicer(p[0:2],p[2])
            x = self.transform_image(img)
            x=x.view(-1, 1, 256, 256).float().to(self.device)
            pred_tensor=self.unet_best(x)
            pred=pred_tensor.view(256, 256)
            pred=pred.cpu().detach().numpy()
            # pred=pred*0.5+0.5
            centroid,num_contours=self.get_image_centroid(pred)
            num_p+=1
            # if sys.version[0]!=2:
            print('\r %f%%\r' % (num_p*100/len(list_IDs)),end='')
            if num_contours>5:
                continue
            elif num_contours>1:
                _,pred_th=cv2.threshold(pred,0.5,1.0,0)
                area=len(np.where(pred_th>0.9)[0])
            elif num_contours==1:
                _,pred_th=cv2.threshold(pred,0.5,1.0,0)
                area=len(np.where(pred_th>0.9)[0])

                if area>self.threshold:
                    pixel_pose=self.find_pixel_pose(centroid[0],p)
                    self.center_line[tuple(pixel_pose)]+=1
                else:
                    continue
        self.center_line=np.array(self.center_line)/np.max(self.center_line)
        
        x_line=[]
        y_line=[]
        for x in range(np.shape(self.center_line)[0]):
            for y in range(np.shape(self.center_line)[1]):
                if self.center_line[x,y]!=0:
                    x_line.append(x)
                    y_line.append(y)
        
        self.model = Pipeline([('poly', PolynomialFeatures(degree=1)),('linear', HuberRegressor(fit_intercept=False))])
        
        self.x_min=min(x_line)
        self.x_max=max(x_line)
        
        self.model = self.model.fit(np.array(x_line).reshape([-1,1]), np.array(y_line))
        self.coef=self.model.named_steps['linear'].coef_
            
        self.find_goal_pos(points_interval)
        
        return self.center_line
    
    def get_parallel_view(self, x):
        y=self.model.predict([[x]])[0]
        gradient=0
        for i in np.arange(1,len(self.coef)):
            gradient+=self.coef[i]*x**(i-1)
        theta=np.arctan(gradient)
        if theta<0:
            theta+=math.pi
        return [int(x),int(y),theta]
    
    def get_vertical_view(self, x):
        
        gradient=0
        for i in np.arange(1,len(self.coef)):
            gradient+=self.coef[i]*x**(i-1)
        theta=np.arctan(gradient)+math.pi/2
        
        y=self.model.predict([[x]])[0]
        offset=(-1)**np.random.randint(2)*np.random.randint(0,50)
        x+=offset*math.sin(theta-math.pi/2)
        y-=offset*math.cos(theta-math.pi/2)
        return [int(x),int(y),theta]
    
    def vessel_existance(self,pos,theta):
        try:
            img,_,_=self.get_slicer(pos,theta)
            x = self.transform_image(img)
            x=x.view(-1, 1, 256, 256).float().to(self.device)
            pred_tensor=self.unet_best(x)
            pred=pred_tensor.view(256, 256)
            pred=pred.cpu().detach().numpy()
            # pred=pred*0.5+0.5
            self.pred=pred
    
            uint_img = np.array(pred*255).astype('uint8')
            ret,thresh = cv2.threshold(uint_img,127,255,0)
            if self.version=='2':
                _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
            areas=[]
            for c in contours:
                area=cv2.contourArea(c)
                areas.append(area)
            # self.max_area=np.max(areas)
            if np.max(areas)<self.threshold:
                return False
            else:
                return True
        except:
            return False
        
        