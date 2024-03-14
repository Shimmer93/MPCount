import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from glob import glob
from PIL import Image
 
import os
import argparse
from multiprocessing import Pool

def gaussian_filter_density(img, points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    #print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 3:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = 15 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    #print ('done.')
    return density

def gaussian_filter_density_fixed(img, points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    #print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        sigma = 4 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, truncate=7/sigma, mode='constant')
    #print ('done.')
    return density

def run(img_fn):
    img_ext = os.path.splitext(img_fn)[1]
    basename = os.path.basename(img_fn).replace(img_ext, '')
    gt_fn = img_fn.replace(img_ext, '.npy')
    dmap_fn = gt_fn.replace(basename, basename + '_dmap')

    if os.path.exists(dmap_fn):
        return

    img = cv2.imread(img_fn)
    gt = np.load(gt_fn)
    dmap = gaussian_filter_density_fixed(img, gt)
    np.save(dmap_fn, dmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        raise Exception("Path does not exist")

    img_fns = []
    for phase in ['train', 'val', 'test']:
        img_fns += glob(os.path.join(path, phase, '*.jpg'))
    new_fns = []
    for fn in img_fns:
        if 'aug' in fn:
            continue
        new_fns.append(fn)
    img_fns = new_fns

    with Pool(8) as p:
        r = list(tqdm(p.imap(run, img_fns), total=len(img_fns)))