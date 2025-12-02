from utils import *
import cv2
import matplotlib.colors as mcolors
import os
from PIL import Image
import open3d as o3d
import numpy as np
import time
import pickle as pkl
from .visualizer import Visualizer
from detectron2.data import MetadataCatalog
from slam.utils import selectPointsFromVoxel,extractVoxelCenterfromVoxelmap
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
import json
css4_colors = mcolors.TABLEAU_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

def read_pkl(path_file):
    output = open(path_file, 'rb')
    res = pkl.load(output)
    return res

def mapObjects_2D(cam_intrinsic, extrinsic_inv,image_height,image_width,points):
    kernel = np.ones((15, 15), np.uint8)
    depth_buffer = np.zeros((image_height,image_width), dtype=np.float32)
    mask = -np.ones((image_height,image_width), dtype=np.int32)
    for j,point in points.items():
        points_homogeneous = np.hstack(((point), np.ones((point.shape[0], 1))))
        points_homogeneous = (extrinsic_inv @ points_homogeneous.T).T[:, :3]
        points_homogeneous[:, [1, 2]] = points_homogeneous[:, [2, 1]]
        points_camera = cam_intrinsic @ points_homogeneous.T
        points_camera = points_camera.T
        z = points_camera[:,-1]
        u = (points_camera[:,0] / z).astype(np.int32)
        v = (image_height - 1 - points_camera[:,1] / z).astype(np.int32)
        
        index = np.where((u>=0)*(u<image_width)*(v>=0)*(v<image_height)*(z>=0))[0]
        index_buffer = np.where((depth_buffer[v[index],u[index]]==0) | (depth_buffer[v[index],u[index]]>z[index]))[0]
        if index[index_buffer].shape[0] > 0:
            mask1 = np.zeros((image_height,image_width), dtype=np.uint8)
            mask1[v[index][index_buffer],u[index][index_buffer]] = 1
            closed_mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
            depth_buffer[np.where(closed_mask!=0)[0],np.where(closed_mask!=0)[1]] = np.mean(z[index][index_buffer])
            mask[np.where(closed_mask!=0)[0],np.where(closed_mask!=0)[1]] = j
    return mask

def mapObjects(cam_intrinsic, extrinsic_inv,image_height,image_width,objects):
    kernel = np.ones((15, 15), np.uint8)
    depth_buffer = np.zeros((image_height,image_width), dtype=np.float32)
    mask = -np.ones((image_height,image_width), dtype=np.int32)
    for j in range(len(objects)):
        point =  extractVoxelCenterfromVoxelmap(objects[j]['voxel_index'])
        points_homogeneous = np.hstack(((point), np.ones((point.shape[0], 1))))
        points_homogeneous = (extrinsic_inv @ points_homogeneous.T).T[:, :3]
        points_homogeneous[:, [1, 2]] = points_homogeneous[:, [2, 1]]
        points_camera = cam_intrinsic @ points_homogeneous.T
        points_camera = points_camera.T
        z = points_camera[:,-1]
        u = (points_camera[:,0] / z).astype(np.int32)
        v = (image_height - 1 - points_camera[:,1] / z).astype(np.int32)
        
        index = np.where((u>=0)*(u<image_width)*(v>=0)*(v<image_height)*(z>=0))[0]
        index_buffer = np.where((depth_buffer[v[index],u[index]]==0) | (depth_buffer[v[index],u[index]]>z[index]))[0]
        if index[index_buffer].shape[0] > 0:
            mask1 = np.zeros((image_height,image_width), dtype=np.uint8)
            mask1[v[index][index_buffer],u[index][index_buffer]] = 1
            closed_mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
            depth_buffer[np.where(closed_mask!=0)[0],np.where(closed_mask!=0)[1]] = np.mean(z[index][index_buffer])
            mask[np.where(closed_mask!=0)[0],np.where(closed_mask!=0)[1]] = j
    return mask

def readImage(path):
    # [修复] 返回 PIL Image 而不是 numpy 数组，方便后续 resize
    image = Image.open(path)
    return image

def dealMask(mask,idxs,image_height,image_width):
    masks=[]
    id=[]
    area=[]
    for i in idxs:
        index = np.where(mask==i)[0]
        if index.shape[0] > 0 :
            mask1=np.zeros((image_height,image_width), dtype=np.uint8)
            mask1[np.where(mask==i)[0],np.where(mask==i)[1]]=1
            masks.append(mask1)
            id.append(i)
            area.append(index.shape[0])
    area=np.array(area)
    index=np.argsort(area)
    return np.array(masks)[index],np.array(id)[index]

def dealMaskReplica(mask,length,image_height,image_width):
    masks=[]
    id=[]
    area=[]
    for i in range(length):
        index = np.where(mask==i)[0]
        if index.shape[0] > 0 :
            mask1=np.zeros((image_height,image_width), dtype=np.uint8)
            mask1[np.where(mask==i)[0],np.where(mask==i)[1]]=1
            masks.append(mask1)
            id.append(i)
            area.append(index.shape[0])
    area=np.array(area)
    index=np.argsort(area)
    return np.array(masks)[index],np.array(id)[index]

def visualIndicated(data):
    point1=data[33]['pcd_np']
    point2=data[36]['pcd_np']
    pcd1=o3d.geometry.PointCloud()
    pcd1.points=o3d.utility.Vector3dVector(point1)
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2=o3d.geometry.PointCloud()
    pcd2.points=o3d.utility.Vector3dVector(point2)
    pcd2.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd2])

def generateMaskMark(image_list,masks,image_origin_dir,output_path,object_length,label_mode='1', alpha=0.3, anno_mode=['Mask','Mark']):
    for i in range(len(image_list)):
        image_ori=readImage(os.path.join(image_origin_dir,image_list[i]))
        visual = Visualizer(np.array(image_ori), metadata=metadata)
        mask_new,id=dealMask(masks[i],object_length)
        for j in range(len(id)):
            demo = visual.draw_binary_mask_with_number(mask_new[j],color=color_proposals[id[j]%len(color_proposals)],text=str(id[j]), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        im = demo.get_image()   
        image=Image.fromarray(im)
        image.save(os.path.join(output_path,image_list[i])) 
        print("save success")
        del visual

def generateProjectMatrix(intrinsic,extrinsic):
    extrinsic =  intrinsic @ extrinsic[:3,:]
    return extrinsic

def generateSoMImage(cfg,image_objects_dict,objects,reses,label_mode='1', alpha=0.3, anno_mode=['Mask','Mark']):
    for img,idxs in image_objects_dict.items():
        res = reses[int(img.split('/')[-1].split('.')[0])]
        image_rgb,cam_K,pose = res[0].cpu().numpy(),res[2][:3,:3],res[3].cpu().numpy()
        visual = Visualizer(image_rgb, metadata=metadata)
        points={}
        for idx in idxs :
            points[idx] = selectPointsFromVoxel(objects[idx])
        mask=mapObjects_2D(cam_K[:3, :3], np.linalg.inv(pose),cfg.screen_h,cfg.screen_w,points)
        mask_new,id=dealMask(mask,points.keys(),cfg.screen_h,cfg.screen_w)
        for j in range(len(id)):
            demo = visual.draw_binary_mask_with_number(mask_new[j],color=color_proposals[id[j]%len(color_proposals)],text=str(id[j]), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        im = demo.get_image()   
        image=Image.fromarray(im)
        image.save(img) 
        del visual
    return True

def generateSomPrompt(cfg, color_path, som_path, pose, cam_K, objects, label_mode='1', alpha=0.3, anno_mode=['Mask','Mark'], img_name=None):
    # [修改] 兼容路径字符串和 PIL Image 对象
    if isinstance(color_path, str):
        image_ori = readImage(color_path)
        # 如果传入是路径，且没有指定 img_name，则从路径中提取
        if img_name is None:
            img_name = str(color_path).split('/')[-1].split('.')[0]
    else:
        image_ori = color_path
        # 如果传入是对象，必须提供 img_name，否则使用默认名防止报错
        if img_name is None:
            img_name = "unknown_idx"

    # [修改] 强制 Resize 到 320x240，防止 OOM 和尺寸不匹配
    #if image_ori.size != (320, 240):
    #    image_ori = image_ori.resize((320, 240))

    # [关键] 转换为 numpy 并确保是 uint8 类型，防止 Visualizer 报错
    image_np = np.asarray(image_ori).astype(np.uint8)
    
    visual = Visualizer(image_np, metadata=metadata)
    
    # 生成 Mask
    mask = mapObjects(cam_K, np.linalg.inv(pose), cfg.env_frame_height, cfg.env_frame_width, objects)
    mask_new, id = dealMaskReplica(mask, len(objects), cfg.env_frame_height, cfg.env_frame_width)
    
    demo = None
    # 如果没有检测到物体，也需要保证 visualizer 正常工作或跳过
    if len(id) > 0:
        for j in range(len(id)):
            demo = visual.draw_binary_mask_with_number(mask_new[j], color=color_proposals[id[j]%len(color_proposals)], text=str(id[j]), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
    
    # 只有在成功绘制了 demo 后才保存
    if demo is not None:
        im = demo.get_image()    
        image = Image.fromarray(im)
        image.save(os.path.join(som_path, img_name + '.png')) 
        del visual
        return True
    return False

def somForMerge(cfg,image_np,pose,cam_K,points,som_path,idx,once,label_mode='1', alpha=0.3, anno_mode=['Mask','Mark']):
    visual = Visualizer(image_np, metadata=metadata)
    mask=mapObjects_2D(cam_K[:3, :3], np.linalg.inv(pose),cfg.image_height,cfg.image_width,points)
    mask_new,id=dealMask(mask,len(points),cfg.image_height,cfg.image_width)
    for j in range(len(id)):
        demo = visual.draw_binary_mask_with_number(mask_new[j],color=color_proposals[id[j]],text=str(id[j]), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
    im = demo.get_image()   
    image=Image.fromarray(im)
    image.save(som_path+str(idx)+"_"+str(once)+".png")  
    del visual
    return True