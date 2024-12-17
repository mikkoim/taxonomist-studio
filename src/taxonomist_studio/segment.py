import argparse
import json

from pathlib import Path
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

from skimage.filters import gaussian, threshold_triangle, unsharp_mask
from skimage import exposure
from skimage.color import rgb2hsv
from scipy import ndimage as ndi

from . import tools

def get_bbox(mask):
    """Calculates a bounding box for a binary mask. The bounding box is calculated for the maximum blob"""
    mask = np.array(mask)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea) 
    x,y,w,h = cv2.boundingRect(cnt)
    return (x,y,w,h)


def segment(img, c):
    """A basic segmentation pipeline of sharpening, histogram eq, blur, thresholding, hole filling and maximum contour finding"""
    if c == 'hsv':
        B = rgb2hsv(np.array(img))[:,:,2]
    else:
        B = np.array(img)[:,:,c]

    B = unsharp_mask(B, radius=2, amount=3)

    B = exposure.equalize_adapthist(B, clip_limit=0.01)

    # Blur
    B_blur = gaussian(B, sigma=1)

    # Threshold 
    thresh = threshold_triangle(B_blur)
    mask = B_blur < thresh
    if mask.sum().sum()/mask.size > 0.5:
        mask = B_blur > thresh


    # Fill holes
    fill = ndi.binary_fill_holes(mask)
    
    # Find contours 
    cnts, _ = cv2.findContours(fill.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea) 
    
    # Select max contour
    out = np.zeros(fill.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(fill.astype(np.uint8), out)

    # Find bounding box
    x,y,w,h = get_bbox(out) 
    return out, (x,y,w,h)

def segment_multi(img):
    """Combines segmetations for a rgb image by their union"""
    out_f = np.zeros_like(np.array(img)[:,:,0]).astype(bool)
    for c in [0,1,2]:
        out, _ = segment(img, 0)
        out_f = (out_f | out.astype(bool)).astype(bool)
    bbox_f = get_bbox(out_f)
    return out_f, bbox_f
    
def does_bbox_hit_border(bbox, img):
    """Returns true if the bounding box hits any of the four borders of img"""
    bbox = xywh2xyxy(bbox)
    if bbox[0] == 0:
        return True
    elif bbox[1] == 0:
        return True
    elif bbox[2] == img.shape[0]:
        return True
    elif bbox[3] == img.shape[1]:
        return True
    return False

def xywh2xyxy(bbox):
    """Converts XYWH bounding box to an XYXY bounding box"""
    return (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
def xyxy2xywh(bbox):
    """Converts XYXY bounding box to an XYWH bounding box"""
    return (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

def coco_fname_anns(coco: COCO, fname):
    """Takes a pytotocools.coco"""
    imgs = []
    for img in coco.imgs.values():
        if fname == img["file_name"]:
            imgs.append(img)
    return imgs

def get_mask_and_bbox_from_coco(coco: COCO, fname):
    """Returns a mask from a COCO annotation"""
    img_coco = coco_fname_anns(coco, fname.name)[0]
    ann = coco.loadAnns(coco.getAnnIds(imgIds=[img_coco["id"]]))[0]
    mask = mask_utils.decode(ann["segmentation"])
    bbox = ann["bbox"]
    return mask, bbox

def get_color_mask(mask):
    """Returns a random color mask from a binary mask"""
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def add_mask(img, mask):
    """Superimposes a binary mask on an image"""
    img = img.copy()
    mask_image = get_color_mask(mask)
    mask_pil = Image.fromarray((mask_image*255).astype(np.uint8))

    # Add the mask_image to the img
    img.paste(mask_pil, (0, 0), mask_pil)
    return img

def add_bbox(img, bbox):
    """Adds a xyxy bounding box to an image"""
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
    rec = cv2.rectangle(img, bbox, (255,0,0), 1)
    return Image.fromarray(rec)

def show_mask(img, mask, ax):
    """ Shows a binary mask on an image
    https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb"""

    ax.imshow(np.array(img))
    mask_image = get_color_mask(mask)
    ax.imshow(mask_image)
    plt.axis("off")

def show_ann(coco: COCO, fname):
    """Displays COCO annotations (mask, bbox), of a filename"""
    img = Image.open(fname)
    mask, bbox = get_mask_bbox(coco, fname)
    ax = plt.gca()
    show_mask(add_bbox(img, bbox), mask, ax)

def run_segment(args):
    """Runs the segmentation pipeline on a BioDiscover csv file and
    saves the output in a COCO format
    
    Args:
        args.data_folder: The folder where the images are stored
        args.csv_path: The path to the BioDiscover csv file
        args.out_folder: The output folder
        args.out_prefix: The prefix of the output files. Suffixes are
                        always _coco_masks.json and _bbox.csv
        args.head: The number of images to process. If None, all images are processed
    
    """

    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.csv_path, sep=";", encoding="ISO-8859-1")
    fnames = tools.load_fpaths(df, args.data_folder, args.species_level)

    print("Done")

    if args.head:
        fnames = fnames[:args.head]

    def parallel_func(f, i):
        img = Image.open(f)
        mask, _ = segment_multi(img)

        mask_rle = mask_utils.encode(np.asfortranarray(mask))
        mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
        bbox = mask_utils.toBbox(mask_rle)
        hits_border = does_bbox_hit_border(bbox, np.array(img))

        image_info = {}
        image_info["id"] = i
        image_info["width"] = img.width
        image_info["height"] = img.height
        image_info["file_name"] = f.name

        annotation = {}
        annotation["id"] = i
        annotation["image_id"] = i
        annotation["segmentation"] = mask_rle
        annotation["bbox"] = bbox.astype(np.uint8).tolist()
        annotation["area"] = int(mask_utils.area(mask_rle))
        annotation["iscrowd"] = False

        return image_info, annotation, bbox, hits_border

    output = Parallel(n_jobs=-1)(delayed(parallel_func)(f,i) for i, f in tqdm(enumerate(fnames), total=len(fnames)))

    out_lists = list(zip(*output))

    coco = {'images': out_lists[0],
            'annotations': out_lists[1]}

    bbox_df = pd.DataFrame({'fname': [f.name for f in fnames],
                            'bbox': out_lists[2],
                            'hits_border': out_lists[3]})

    with open(out_folder / f"{args.out_prefix}_coco_masks.json", "w",
              encoding="utf-8") as f:
        json.dump(coco, f)

    bbox_df.to_csv(out_folder / f"{args.out_prefix}_bbox.csv", index=False)
    print("Output in", out_folder.resolve())