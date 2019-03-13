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
#照片文件2的根目录
PICTURE2_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\image_2')
#照片文件3的根目录
PICTURE3_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\image_3')
#雷达文件的根目录
LIDAR_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\\velodyne')
#标定文件的根目录
CALIB_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\calib')
#LABLE文件的根目录
LABEL_DIR=os.path.join(kitti_object.KITTI_DIR,'object\\training\label_2')
print(LIDAR_DIR)


'''
#测试内容
#尝试画出点云数据
#尝试画出图片
print(PICTURE2_DIR)
filename_dir=os.path.join(PICTURE2_DIR,'000000.png')
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

#读取点云数据
Lidar_file_dir=os.path.join(LIDAR_DIR,'000003.bin')
scan = np.fromfile(Lidar_file_dir, dtype=np.float32)
scan = scan.reshape((-1, 4))

#尝试读取标定信息

calib_file_dir=os.path.join(CALIB_DIR,'000003.txt')
calib_data={}
with open(calib_file_dir,'r') as rf:
    for line in rf.readlines():
        if len(line)<=1:
            continue
        line=line.rstrip(" ")
        key,value=line.split(':',1)
        calib_data[key]=np.array([float(x) for x in value.split()])
 
P2=calib_data['P2']
P2=np.reshape(P2,[3,4])

P3=calib_data['P3']
P3=np.reshape(P3,[3,4])

R0_rect=calib_data['R0_rect']
R0_rect=np.reshape(R0_rect,[3,3])

Tr_velo_to_cam=calib_data['Tr_velo_to_cam']
Tr_velo_to_cam=np.reshape(Tr_velo_to_cam,[3,4])


#尝试读取label文件
label_file_dir=os.path.join(LABEL_DIR,'000003.txt')
object_3d=[]
with open(label_file_dir,'r') as rf:
    for line in rf.readlines():
        if len(line)==0:
            continue
        object_3d.append(kitti_object.Object3d(line))
#尝试写投影的函数
#代码从frustum-pointnets中复制来的


def project_velo_to_ref(pts_3d_velo):                #把雷达坐标系下的点 投影到参考相机坐标系（ref坐标系）中,输入是 3*4
    return np.dot(pts_3d_velo, np.transpose(Tr_velo_to_cam))#self.V2C (3,4)  ,输出是 n*3

def project_ref_to_rect(pts_3d_ref):
   #  Input and Output are nx3 points 
    return np.transpose(np.dot(R0_rect, np.transpose(pts_3d_ref)))


def project_rect_to_image( pts_3d_rect,PX):
   #    Input: nx3 points in rect camera coord.
    #    Output: nx2 points in image2 coord.
    #
    n = pts_3d_rect.shape[0]
    pts_3d_rect=np.hstack((pts_3d_rect , np.ones((n,1))))   #增加一个维度
    pts_2d = np.dot(pts_3d_rect, np.transpose(PX)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

#过滤掉不在2D框中的点云数据
def extract_pts_from_2D(pts_2D,xmin,ymin,xmax,ymax):   #pts_2D是 n*2的,返回掩模
    fov_inds =(pts_2D[:,0] >= xmin) & (pts_2D[:,0]<=xmax )& ( pts_2D[:,1]>=ymin ) & ( pts_2D[:,1]<=ymax)
    return fov_inds

#把点云投影到ref坐标系上
pts_in_ref=project_velo_to_ref(scan)

#把点云投影到rect坐标系上
pts_in_rect=project_ref_to_rect(pts_in_ref)

#把点云投影到 iamge2坐标系上

pts_in_image_2D_2=project_rect_to_image(pts_in_rect,P2)
pts_in_image_2D_3=project_rect_to_image(pts_in_rect,P3)

fov_2d_inds_2=extract_pts_from_2D(pts_in_image_2D_2, object_3d[0].xmin, object_3d[0].ymin, object_3d[0].xmax, object_3d[0].ymax)

pts_fov_in_rect=pts_in_rect[fov_2d_inds_2,:]

#把过滤出来的点云投影到第二张照片上
pts_fov_in_image3=project_rect_to_image(pts_fov_in_rect,P3)
#找到这些点云的边界

xmin3=pts_fov_in_image3[0,0] ;ymin3=pts_fov_in_image3[0,1];xmax3=pts_fov_in_image3[0,0];ymax3=pts_fov_in_image3[0,1]
for p in pts_fov_in_image3:
    if p[0]<xmin3:
        xmin3=p[0]
    if p[0]>xmax3:
        xmax3=p[0]
    if p[1]<ymin3:
        ymin3=p[1]
    if p[1]>ymax3:
        ymax3=p[1]
print(xmin3,ymin3,xmax3,ymax3)    
#进行修正 
xmax3=xmin3+(object_3d[0].xmax-object_3d[0].xmin)

fov_2d_inds_3=extract_pts_from_2D(pts_in_image_2D_3,xmin3,ymin3,xmax3,ymax3)
fov_2d_inds=fov_2d_inds_2 & fov_2d_inds_3

pts_fov2=scan[fov_2d_inds_2,:]
pts_fov=scan[fov_2d_inds,:]

fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),fgcolor=None, engine=None, size=(500, 500))
mlab.points3d(pts_fov2[:,0], pts_fov2[:,1], pts_fov2[:,2], pts_fov2[:,1], mode='point',    colormap='gnuplot', scale_factor=1, figure=fig)
mlab.show()
'''
filename3_dir=os.path.join(PICTURE3_DIR,'000003.png')
filename2_dir=os.path.join(PICTURE2_DIR,'000003.png')
picture2=cv2.imread(filename2_dir)
picture3=cv2.imread(filename3_dir)
cv2.namedWindow('picture2',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('picture3',cv2.WINDOW_AUTOSIZE)
cv2.rectangle(picture2, ( int(object_3d[0].xmin),int(object_3d[0].ymin) ),( int(object_3d[0].xmax) ,int(object_3d[0].ymax)) ,(0,255,0))
cv2.rectangle(picture3,(int(xmin3),int(ymin3)),(int(xmax3),int(ymax3)),(0,255,0))
cv2.imshow('picture2',picture2)
cv2.imshow('picture3',picture3)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

            
