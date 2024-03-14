from PIL import Image
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import cv2
import argparse
from random import shuffle
from scipy.io import loadmat

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*point, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data_jhu(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('images', 'gt').replace('.jpg', '.txt')
    points = []
    with open (mat_path, 'r') as f:
        while True:
            point = f.readline()
            if not point:
                break
            point = point.split(' ')[:-1]
            points.append([float(point[0]), float(point[1])])
    points = np.array(points)
    if len(points>0):
        idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
        points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def generate_data_qnrf(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '_ann.mat')
    points = loadmat(mat_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def generate_data_smartcity(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '.mat')
    points = loadmat(mat_path)['loc'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def generate_data_sta(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    name = im_path.split('/')[-1].split('.')[0]
    mat_path = os.path.abspath(os.path.join(im_path, os.pardir, os.pardir, 'ground-truth', 'GT_' + name + '.mat'))
    points = loadmat(mat_path)['image_info'][0][0][0][0][0].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def generate_data_cc50(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '_ann.mat')
    points = loadmat(mat_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def generate_data_fdst(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    name = im_path.split('/')[-1].split('.')[0]
    mat_path = os.path.abspath(os.path.join(im_path, os.pardir, os.pardir, 'annotation', name + '.mat'))
    points = loadmat(mat_path)['annotation'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def generate_data_vidcrowd(im_path, mat_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    points = loadmat(mat_path)['annotation'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    
    return im, points

def generate_data_nwpu(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    name = im_path.split('/')[-1].split('.')[0]
    mat_path = os.path.abspath(os.path.join(im_path, os.pardir, os.pardir, 'mats', name + '.mat'))
    if os.path.exists(mat_path):
        points = loadmat(mat_path)['annPoints'].astype(np.float32)
        if len(points) > 0:
            idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
            points = points[idx_mask]
    else: 
        points = None
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        if points is not None:
            points = points * rr
    return Image.fromarray(im).convert('RGB'), points

def generate_data_worldexpo(im_path, mat_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    points = loadmat(mat_path)['annotation'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    
    return im, points

def generate_data_mall(im_path, points, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def generate_data_ucsd(im_path, points, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def run_jhu(origin_dir, save_dir, min_size, max_size):
    for phase in ['train', 'val', 'test']:
            sub_dir = os.path.join(origin_dir, phase)
            sub_save_dir = os.path.join(save_dir, phase)
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_list = glob(os.path.join(os.path.join(sub_dir, 'images'), '*jpg'))
            for im_path in tqdm(im_list):
                name = os.path.basename(im_path)
                im, points = generate_data_jhu(im_path, min_size, max_size)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path, quality=95)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)

def run_qnrf(origin_dir, save_dir, min_size, max_size):
    for phase in ['Train', 'Test']:
        sub_dir = os.path.join(origin_dir, phase)
        if phase == 'Train':
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(save_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                with open('ucf_{}.txt'.format(sub_phase)) as f:
                    for i in f:
                        im_path = os.path.join(sub_dir, i.strip())
                        name = os.path.basename(im_path)
                        im, points = generate_data_qnrf(im_path, min_size, max_size)
                        im_save_path = os.path.join(sub_save_dir, name)
                        im.save(im_save_path)
                        gd_save_path = im_save_path.replace('jpg', 'npy')
                        np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for im_path in tqdm(im_list):
                name = os.path.basename(im_path)
                im, points = generate_data_qnrf(im_path, min_size, max_size)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)

def run_smartcity(origin_dir, save_dir, min_size, max_size):
    data_dir = os.path.join(origin_dir, 'images')
    im_list = glob(os.path.join(data_dir, '*jpg'))
    shuffle(im_list)
    train_split = int(len(im_list) * 0.6)
    val_split = int(len(im_list) * 0.8)
    train_list = im_list[:train_split]
    val_list = im_list[train_split:val_split]
    test_list = im_list[val_split:]

    for phase, im_list in zip(['train', 'val', 'test'], [train_list, val_list, test_list]):
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        for im_path in tqdm(im_list):
            name = os.path.basename(im_path)
            im, points = generate_data_smartcity(im_path, min_size, max_size)
            im_save_path = os.path.join(sub_save_dir, name)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)

def run_sta(origin_dir, save_dir, min_size, max_size):
    if origin_dir.split('/')[-1] == 'part_A':
        part = 'sta'
    else:
        part = 'stb'
    for phase in ['train', 'test']:
        sub_dir = os.path.join(origin_dir, phase+'_data')
        if phase == 'train':
            sub_phase_list = ['train', 'val']
            im_path = os.path.join(sub_dir, 'images')
            im_list = glob(os.path.join(im_path, '*jpg'))
            with open(f'{part}_train.txt') as f:
                train_list = f.readlines()
            with open(f'{part}_val.txt') as f:
                val_list = f.readlines()
            train_list = [os.path.join(im_path, i.strip()) for i in train_list]
            val_list = [os.path.join(im_path, i.strip()) for i in val_list]
            for sub_phase, im_list in zip(sub_phase_list, [train_list, val_list]):
                sub_save_dir = os.path.join(save_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                for im_path in tqdm(im_list):
                    name = os.path.basename(im_path)
                    im, points = generate_data_sta(im_path, min_size, max_size)
                    im_save_path = os.path.join(sub_save_dir, name)
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_path = os.path.join(sub_dir, 'images')
            im_list = glob(os.path.join(im_path, '*jpg'))
            for im_path in tqdm(im_list):
                name = os.path.basename(im_path)
                im, points = generate_data_sta(im_path, min_size, max_size)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)

def run_cc50(origin_dir, save_dir, min_size, max_size):
    im_list = glob(os.path.join(origin_dir, '*jpg'))
    shuffle(im_list)
    train_split = int(len(im_list) * 0.6)
    val_split = int(len(im_list) * 0.8)
    train_list = im_list[:train_split]
    val_list = im_list[train_split:val_split]
    test_list = im_list[val_split:]

    for phase, im_list in zip(['train', 'val', 'test'], [train_list, val_list, test_list]):
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        for im_path in tqdm(im_list):
            name = os.path.basename(im_path)
            im, points = generate_data_cc50(im_path, min_size, max_size)
            im_save_path = os.path.join(sub_save_dir, name)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)

def run_fdst(origin_dir, save_dir, min_size, max_size):
    for phase in ['train', 'test']:
        sub_dir = os.path.join(origin_dir, phase)
        if phase == 'train':
            sub_phase_list = ['train', 'val']
            im_path = os.path.join(sub_dir, 'img')
            im_list = glob(os.path.join(im_path, '*jpg'))
            # video_list = [1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,
            #               31,32,33,36,37,38,41,42,43,46,47,48,51,52,53,56,57,58,
            #               61,62,63,66,67,68,71,72,73,76,77,78,81,82,83,86,87,88,
            #               91,92,93,96,97,98]
            # shuffle(video_list)
            # train_list = video_list[:50]
            train_list = [1,2,6,7,11,12,16,17,21,22,26,27,
                          31,32,36,37,41,42,46,47,51,52,56,57,
                          61,62,66,67,71,72,76,77,81,82,86,87,
                          91,92,96,97]
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(save_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
            for im_path in tqdm(im_list):
                name = os.path.basename(im_path)
                video_id = int(name.split('_')[0])
                if video_id in train_list:
                    sub_phase = 'train'
                else:
                    sub_phase = 'val'
                sub_save_dir = os.path.join(save_dir, sub_phase)
                im_save_path = os.path.join(sub_save_dir, name)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                if os.path.exists(im_save_path) and os.path.exists(gd_save_path):
                    continue
                im, points = generate_data_fdst(im_path, min_size, max_size)
                im.save(im_save_path)
                np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_path = os.path.join(sub_dir, 'img')
            im_list = glob(os.path.join(im_path, '*jpg'))
            for im_path in tqdm(im_list):
                name = os.path.basename(im_path)
                im_save_path = os.path.join(sub_save_dir, name)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                if os.path.exists(im_save_path) and os.path.exists(gd_save_path):
                    continue
                im, points = generate_data_fdst(im_path, min_size, max_size)
                im.save(im_save_path)
                np.save(gd_save_path, points)

def run_vidcrowd(origin_dir, save_dir, min_size, max_size):
    for phase in ['train', 'test']:
        sub_dir = os.path.join(origin_dir, f'VidCrowd_{phase}_ann_newsplit')
        gt_list = glob(os.path.join(sub_dir, '*.mat'))
        if phase == 'train':
            val_video_list = ['10', '12', '13', '16']
            val_gt_list = []
            for val_video in val_video_list:
                val_gt_list += glob(os.path.join(sub_dir, val_video+'_*.mat'))
            train_gt_list = list(set(gt_list) - set(val_gt_list))
            sub_phase_list = ['train', 'val']
            for sub_phase, sub_gt_list in zip(sub_phase_list, [train_gt_list, val_gt_list]):
                sub_save_dir = os.path.join(save_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                for gt_path in tqdm(sub_gt_list):
                    name = os.path.basename(gt_path)
                    name = name.replace('mat', 'jpg')
                    im_path = os.path.join(origin_dir, 'images', name)
                    im, points = generate_data_vidcrowd(im_path, gt_path, min_size, max_size)
                    im_save_path = os.path.join(sub_save_dir, name)
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            for gt_path in tqdm(gt_list):
                name = os.path.basename(gt_path)
                name = name.replace('mat', 'jpg')
                im_path = os.path.join(origin_dir, 'images', name)
                im, points = generate_data_vidcrowd(im_path, gt_path, min_size, max_size)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)

def run_nwpu(origin_dir, save_dir, min_size, max_size):
    img_dir = os.path.join(origin_dir, 'images')
    img_fns = glob(os.path.join(img_dir, '*.jpg'))
    for phase in ['train', 'val', 'test']:
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)

    for img_fn in tqdm(img_fns):
        name = os.path.basename(img_fn).split('.')[0]
        if int(name) <= 3109:
            phase = 'train'
        elif int(name) <= 3609:
            phase = 'val'
        else:
            phase = 'test'
        sub_save_dir = os.path.join(save_dir, phase)
        im, points = generate_data_nwpu(img_fn, min_size, max_size)
        im_save_path = os.path.join(sub_save_dir, name+'.jpg')
        im.save(im_save_path)
        if not points is None:
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)

def run_mall(origin_dir, save_dir, min_size, max_size):
    img_dir = os.path.join(origin_dir, 'frames')
    gt_fn = os.path.join(origin_dir, 'mall_gt.mat')
    im_list = sorted(glob(os.path.join(img_dir, '*jpg')))
    train_split = 600
    val_split = 800
    train_list = im_list[:train_split]
    val_list = im_list[train_split:val_split]
    test_list = im_list[val_split:]

    pts_list = loadmat(gt_fn)['frame'][0]

    idx = 0
    for phase, im_list in zip(['train', 'val', 'test'], [train_list, val_list, test_list]):
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        for im_path in tqdm(im_list):
            name = os.path.basename(im_path)
            pts = pts_list[idx][0][0][0].astype(np.float32)
            im, points = generate_data_mall(im_path, pts, min_size, max_size)
            im_save_path = os.path.join(sub_save_dir, name)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)
            idx += 1

def run_ucsd(origin_dir, save_dir, min_size, max_size):
    trainval_list = ['003', '004', '005', '006']
    test_list = ['000', '001', '002', '007', '008', '009']
    for phase, sub_list in zip(['trainval', 'test'], [trainval_list, test_list]):
        if phase == 'trainval':
            train_save_dir = os.path.join(save_dir, 'train')
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir)
            val_save_dir = os.path.join(save_dir, 'val')
            if not os.path.exists(val_save_dir):
                os.makedirs(val_save_dir)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
        for sub in sub_list:
            sub_dir = os.path.join(origin_dir, 'video', 'vidf', f'vidf1_33_{sub}.y')
            im_list = sorted(glob(os.path.join(sub_dir, '*png')))
            gt_fn = os.path.join(origin_dir, 'gt', 'vidf', f'vidf1_33_{sub}_frame_full.mat')
            pts_list = loadmat(gt_fn)['fgt'][0][0][0][0]
            for idx, im_path in tqdm(enumerate(im_list)):
                name = os.path.basename(im_path)
                pts = pts_list[idx][0][0][0].astype(np.float32)
                pts = pts[:, :2]
                im, points = generate_data_ucsd(im_path, pts, min_size, max_size)
                if phase == 'trainval':
                    if idx < 180:
                        sub_save_dir = train_save_dir
                    else:
                        sub_save_dir = val_save_dir
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('png', 'npy')
                np.save(gd_save_path, points)

def run_ucsd2(origin_dir, save_dir, min_size, max_size):
    train_list = ['003', '004', '005']
    val_list = ['006']
    test_list = ['000', '001', '002', '007', '008', '009']
    for phase, sub_list in zip(['train', 'val', 'test'], [train_list, val_list, test_list]):
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        for sub in sub_list:
            sub_dir = os.path.join(origin_dir, 'video', 'vidf', f'vidf1_33_{sub}.y')
            im_list = sorted(glob(os.path.join(sub_dir, '*png')))
            gt_fn = os.path.join(origin_dir, 'gt', 'vidf', f'vidf1_33_{sub}_frame_full.mat')
            pts_list = loadmat(gt_fn)['fgt'][0][0][0][0]
            for idx, im_path in tqdm(enumerate(im_list)):
                name = os.path.basename(im_path)
                pts = pts_list[idx][0][0][0].astype(np.float32)
                pts = pts[:, :2]
                im, points = generate_data_ucsd(im_path, pts, min_size, max_size)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('png', 'npy')
                np.save(gd_save_path, points)

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='E:\Dataset\Counting\jhu_crowd_v2.0',
                        help='original data directory')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='processed data directory')
    parser.add_argument('--min-size', default=512, type=int,
                        help='minimum image size')
    parser.add_argument('--max-size', default=2048, type=int,
                        help='maximum image size')
    parser.add_argument('--dataset', default='jhu', type=str, 
                        choices=['jhu', 'qnrf', 'smartcity', 'sta', 'stb', 'cc50', 'fdst', 'vidcrowd', 'worldexpo', 'nwpu', 'mall', 'ucsd'],
                        help='dataset name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    origin_dir = args.origin_dir
    save_dir = args.data_dir
    min_size = args.min_size
    max_size = args.max_size
    dataset = args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if dataset == 'jhu':
        run_jhu(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'qnrf':
        run_qnrf(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'smartcity':
        run_smartcity(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'sta' or dataset == 'stb':
        run_sta(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'cc50':
        run_cc50(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'fdst':
        run_fdst(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'vidcrowd':
        run_vidcrowd(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'nwpu':
        run_nwpu(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'mall':
        run_mall(origin_dir, save_dir, min_size, max_size)
    elif dataset == 'ucsd':
        run_ucsd(origin_dir, save_dir, min_size, max_size)
    else:
        raise NotImplementedError
