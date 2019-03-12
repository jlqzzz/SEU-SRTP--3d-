import os
import sys
import cv2
import kitti_object as kitti_object
import numpy as np
from mayavi import mlab

#prepare预处理模块的文件的根目录，即 prepare_data文件夹的目录
PREPARE_DIR=os.path.dirname(os.path.abspath(__file__))
#项目文件的根目录
ROOT_DIR=os.path.dirname(PREPARE_DIR)
#照片文件的根目录
PICTURE_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\image_2')
#雷达文件的根目录
LIDAR_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\\velodyne')
#标定文件的根目录
CALIB_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\calib')
print(LIDAR_DIR)


'''
#测试内容
#尝试画出点云数据
#尝试画出图片
print(PICTURE_DIR)
filename_dir=os.path.join(PICTURE_DIR,'000000.png')
print(filename_dir)
picture=cv2.imread(filename_dir)
cv2.namedWindow('picture',cv2.WINDOW_AUTOSIZE)
cv2.imshow('picture',picture)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
#尝试绘制出点云数据
Lidar_file_dir=os.path.join(LIDAR_DIR,'000000.bin')
scan = np.fromfile(Lidar_file_dir, dtype=np.float32)
scan = scan.reshape((-1, 4))
fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),fgcolor=None, engine=None, size=(500, 500))
mlab.points3d(scan[:,0], scan[:,1], scan[:,2], scan[:,1], mode='point',    colormap='gnuplot', scale_factor=1, figure=fig)
mlab.show()
'''

#尝试读取标定信息
'''
calib_file_dir=os.path.join(CALIB_DIR,'000000.txt')
calib_data={}
with open(calib_file_dir,'r') as rf:
    for line in rf.readlines():
        if len(line)==0:
            continue
        line=line.rstrip(" ")
        key,value=line.split(':',1)
        calib_data[key]=np.array([float(x) for x in value.split()])
 
P2=calib_data['P2']

P2=np.reshape(P2,[3,4])

R0_rect=calib_data['R0_rect']
R0_rect=np.reshape(R0_rect,[3,3])

Tr_velo_to_cam=calib_data['Tr_velo_to_cam']
Tr_velo_to_cam=np.reshape(Tr_velo_to_cam,[3,4])

print(Tr_velo_to_cam)
'''

#尝试写投影的函数


