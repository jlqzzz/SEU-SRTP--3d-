import os
import sys
import cv2
import kitti_object as kitti_object

#prepare预处理模块的文件的根目录，即 prepare_data文件夹的目录
PREPARE_DIR=os.path.dirname(os.path.abspath(__file__))
#项目文件的根目录
ROOT_DIR=os.path.dirname(PREPARE_DIR)


#测试内容
#尝试画出点云数据
#尝试画出图片
from mayavi import mlab
PICTURE_DIR=os.path.join(kitti_object.KITTI_DIR,'object\training\image_2')
filenameP=os.path.join(PICTURE_DIR,'000000.png')
picture=cv2.imread(filenameP)
cv2.namedWindow('picture',cv2.WINDOW_AUTOSIZE)
cv2.imshow('picture',picture)
cv2.waitKey(0)
cv2.destroyAllWindows()