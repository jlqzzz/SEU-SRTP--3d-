import os
import sys
import pickle as pickle
import numpy as np 

PREPARE_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT_DIR=os.path.dirname(PREPARE_DIR)
KITTI_DIR=os.path.join(ROOT_DIR,'dataset\\kitti')
'''
def class kitti_object(object):
    def __init__( self , keyword='training', dir=KITTI_DIR ):
'''
class Object3d(object): #3D物体类标签定义 -H
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')  #以空格分隔 -H  ps:返回分割后的所有的子字符串，每一个label_file_line就是一个目标 eg： Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01 -Y

        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion //提取标签，截断，遮挡 —H
        self.type = data[0] # 'Car', 'Pedestrian', ...//物体种类
        self.truncation = data[1] # truncated pixel ratio [0..1]  截断程度 h
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown 被遮挡程度 h
        self.alpha = data[3] # object observation angle [-pi..pi] 观察角度 h

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # rotation_y yaw angle (around Y-axis in camera coordinates) [-pi..pi]  
