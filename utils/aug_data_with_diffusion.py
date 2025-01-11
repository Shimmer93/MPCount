import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from scipy.spatial import KDTree
from glob import glob
import os
from argparse import ArgumentParser
from tqdm import tqdm

default_prompt = "realistic photo, highly detailed, high resolution, 8k"
default_negative_prompt = (
    "heads, faces, back of heads, small heads, inconsistent heads, distant crowds, "
    "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, "
    "mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"
)

def points_to_mask(pts, img_shape, k=4, leafsize=2048, default_radius=15, radius_scale=0.5, radius_offset=5):
    w, h = img_shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if len(pts) == 0:
        return mask
    tree = KDTree(pts, leafsize=leafsize)
    distances, _ = tree.query(pts, k=k)
    for pt, dist in zip(pts, distances):
        if pt[0] < 0 or pt[0] >= w or pt[1] < 0 or pt[1] >= h:
            continue
        if len(pts) < k:
            radius = default_radius
        else:
            radius = int(dist[-1] * radius_scale) + radius_offset
        cv2.circle(mask, (int(pt[0]), int(pt[1])), radius, 0, -1)

    return mask
    
def process_img(pipe, img_fn, prompt, negative_prompt, k=4, leafsize=2048, default_radius=15, radius_scale=0.5, radius_offset=5, radius_blur=-1):
    img = load_image(img_fn)
    pts = np.load(img_fn.replace(".jpg", ".npy"))
    mask = Image.fromarray(points_to_mask(pts, img.size, k, leafsize, default_radius, radius_scale, radius_offset))
    if radius_blur < 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=radius_blur))
    
    w, h = img.size
    new_w = (w // 16 + 1) * 16
    new_h = (h // 16 + 1) * 16

    # img = img.crop((0, 0, new_w, new_h))
    # mask = mask.crop((0, 0, new_w, new_h))
    img = ImageOps.expand(img, (0, 0, new_w - w, new_h - h), fill=0)
    mask = ImageOps.expand(mask, (0, 0, new_w - w, new_h - h), fill=0)

    img_new = pipe(width=new_w, height=new_h, prompt=prompt, negative_prompt=negative_prompt, image=img, mask_image=mask).images[0]
    # img_new = ImageOps.expand(img_new, (0, 0, w - new_w, h - new_h), fill=0)
    img_new = img_new.crop((0, 0, w, h))
    
    return img_new

def main(args):
    pipeline = AutoPipelineForInpainting.from_pretrained(args.pretrained_model, torch_dtype=torch.float16)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "test"), exist_ok=True)

    img_fns = glob(os.path.join(args.img_dir, "train", "*.jpg")) + \
              glob(os.path.join(args.img_dir, "val", "*.jpg")) + \
              glob(os.path.join(args.img_dir, "test", "*.jpg"))
    
    for img_fn in tqdm(img_fns):
        for i in range(args.num_aug):
            img_new_fn = img_fn.replace(".jpg", f"_aug_{i}.jpg").replace(args.img_dir, args.out_dir)
            if os.path.exists(img_new_fn):
                continue
            try:
                img_new = process_img(pipeline, img_fn, args.prompt, args.negative_prompt, 
                                    args.k, args.leafsize, args.default_radius, args.radius_scale, args.radius_offset, args.radius_blur)
                img_new.save(img_new_fn)
            except Exception as e:
                print(f"Error processing {img_fn}: {e}")
                continue

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=default_prompt)
    parser.add_argument("--negative_prompt", type=str, default=default_negative_prompt)
    parser.add_argument("--num_aug", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--leafsize", type=int, default=2048)
    parser.add_argument("--default_radius", type=int, default=15)
    parser.add_argument("--radius_scale", type=float, default=0.5)
    parser.add_argument("--radius_offset", type=int, default=5)
    parser.add_argument("--radius_blur", type=int, default=-1)
    args = parser.parse_args()

    main(args)
