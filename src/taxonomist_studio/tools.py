from pathlib import Path
import pandas as pd
import numpy as np
import copy
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import struct
import os
from datetime import datetime

from . import segment

def parse_biodiscover_imagefname(fname):
    date_ext = '_'.join(fname.split('_')[-3:])
    cam_sample_number = fname[:-(len(date_ext)+1)]
    camera = cam_sample_number.split('_')[0]
    run = '_'.join(cam_sample_number.split('_')[1:])
    individual = '_'.join(run.split('_')[:-1])
    number = run.split('_')[-1]

    date = datetime.strptime(date_ext[:-4], "%Y_%m_%d-%H-%M-%S-%f")
    
    p = {'camera': camera,
         'individual': individual,
         'run': run,
         'run_number': number,
         'datetime': date,
         'unixtime': datetime.timestamp(date)}
    return p

def load_fpaths(df, data_folder, species_level):
    if species_level:
        fpaths = df.apply(
            lambda x: Path(
                data_folder,
                *Path(x["Image Path"]).parts[-3:],
                x["Image File Name"],
            ),
            axis=1,
        )
    else:
        fpaths = df["Image File Name"].apply(
            lambda x: get_biodiscover_fpath(data_folder, x)
        )
    return fpaths

def group_features(df, feature_df, group_col):
    """Calculates a mean feature vector for a group in the dataframe.
    
    Args:
        df (pd.DataFrame): BioDiscover metadata dataframe.
        feature_df (pd.DataFrame): Dataframe with features. Produced by taxonomist-studio embedder
        group_col (str): Column name from df to group by.
    Returns:
        df (pd.DataFrame): Dataframe with the mean feature vector for each group.
                        Index is the group name.
    """

    group_idx = df.set_index("Image File Name").loc[feature_df.index][group_col]
    relabeled_features = feature_df.copy()
    relabeled_features.index = group_idx.values
    return relabeled_features.groupby(level=0).mean()

def add_parsed_columns(df):
    """Parses some of the BioDiscover spreadsheet columns and adds them
    as new columns
    """

    ff = df['Image File Name'].apply(parse_biodiscover_imagefname).values

    # Add new columns
    df = pd.concat((df, pd.DataFrame.from_records(ff)), axis=1)
    return df


def get_biodiscover_fpath(data_folder: str, fname: str):
    """Returns the full path to a BioDiscover image file
    
    Args:
        data_folder (str): Path to the folder the data is, usually the Expo_2000_Ap_8 
                            (or similar) folder
        fname (str): Filename
    
    Returns:
        Path: Full path to the file
    """
    p = parse_biodiscover_imagefname(fname)
    fpath = Path(data_folder,
                    p['individual'],
                    p['run'],
                    fname)
    return fpath

def add_additional_biodiscover_columns(df):
    """Adds additional columns to the BioDiscover dataframe
    
    Args:
        df (pd.DataFrame): BioDiscover dataframe
    
    Returns:
        pd.DataFrame: Dataframe with additional columns
    """
    # Parse the filename
    ff = df['Image File Name'].apply(parse_biodiscover_imagefname).values

    # Add new columns
    df = pd.concat((df, pd.DataFrame.from_records(ff)), axis=1)

    # Add other columns
    df["run_camera"] = df.apply(lambda x: f"{x['run']}_cam{x['camera']}", axis=1)


    # Set dtypes
    df = df.astype({'individual': 'str'})
    return df


def calculate_falling_speed(df, top_to_bottom=True, return_grouped=False):
    if top_to_bottom:
        roi_col = "ROI (bottom)"
    else:
        roi_col = "ROI (right)"
        
    if 'individual' not in df.columns:
        df = add_parsed_columns(df)

    grouped = df.groupby(['run', 'camera'])

    # Calculate falling speed
    pos_min = grouped[roi_col].min()
    pos_max = grouped[roi_col].max()
    n_images = grouped[roi_col].count()
    speed = (pos_max-pos_min)/n_images

    return df.merge(speed.rename("speed").reset_index(), on=["run", "camera"])


def biodiscover_to_features(df):

    if "individual" not in df.columns:
        df = add_parsed_columns(df)

    assert np.all(df['Sample Name/Number']==df['sample'])
    df = df.drop(columns=['Sample Name/Number'])

    df = df.astype({'individual': 'str'})

    # Group
    grouped = df.groupby(['sample', 'camera'])

    # Calculate falling speed
    pos_min = grouped['ROI (bottom)'].min()
    pos_max = grouped['ROI (bottom)'].max()
    n_images = grouped['ROI (bottom)'].count()
    speed = (pos_max-pos_min)/n_images

    features = (
            grouped
               .agg({'Area': ['mean','std']})
               .assign(speed=speed)
               .groupby('sample')
               .mean() # mean across cameras
              )

    # Drop values where speed is 0
    features = features[features.speed != 0]

    # Assign columns
    features.columns = ['area_mean', 'area_std', 'speed']
    features = features.assign(individual=grouped.first().individual.groupby('sample').first())
    
    features = features.assign(log_area_mean = np.log(features.area_mean),
                                log_area_std = np.log(features.area_std),
                               log_speed = np.log(features.speed))
                               
    return features


# Image viewing tools

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core.
    By: https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
    """
    size = os.path.getsize(file_path)

    with open(file_path, "rb") as input:
        height = -1
        width = -1
        data = input.read(25)
        if ((size >= 24) and data.startswith(b"\x89PNG")
              and (data[12:16] == b"IHDR")):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
    return width, height

def make_grid(batch, nrow=8):
    """ Numpy-equivalent of the torch make_grid function 
    """
    if isinstance(batch, list):
        batch = np.asarray(batch)
   
    b, h, w, c = batch.shape
    ncol = b // nrow

    img_grid = (batch.reshape(nrow, ncol, h, w, c)
              .swapaxes(1,2)
              .reshape(h*nrow, w*ncol, 3))
    
    return img_grid

def get_image_batch(fname_list,
                    asarray=True,
                    contrast=1.0,
                    brightness=1.0):

    batch = []
    for fname in fname_list:
        # Load and resize
        img = Image.open(fname)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        if asarray:
            I = np.asarray(img)
        else:
            I = img
        batch.append(I)
    return batch

def show_files(fpath_list, 
               label_list=None, 
               nrow=8, 
               img_size=128,
               label_fill=(255,255,255),
               return_img = False,
               autocontrast=False,
               segmentation_coco=None):
    """Displays a batch of filenames
    """
    if not label_list:
        label_list = [""]*len(fpath_list)

    batch = []
    for fname, label in zip(fpath_list,label_list):
        # Load and resize
        img = Image.open(fname)

        if autocontrast:
            img = ImageOps.autocontrast(img, preserve_tone=True)

        # Add possible segmentation mask
        if segmentation_coco:
            mask, bbox = segment.get_mask_and_bbox_from_coco(segmentation_coco, fname)
            img_bbox = segment.add_bbox(img, bbox)
            img = segment.add_mask(img_bbox, mask)

        img = ImageOps.pad(img, (img_size,img_size))
        
        # Draw possible caption
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), str(label), fill=label_fill)
        
        I = np.asarray(img)
        batch.append(I)

    # Add empty patches if row count and image count don't match
    n_zeros = nrow - len(fpath_list) % nrow
    if n_zeros != nrow:
        for _ in range(n_zeros):
            batch.append(np.zeros((img_size,img_size,3), np.uint8))
            
    # Reshape to a single image
    img = Image.fromarray(make_grid(batch, nrow=nrow))
    if return_img:
        return img
    else:
        img.show()


def get_label_list(df, label_col):
    """Concatenates the labels from multiple columns to a single string

    Example:
        label_col = ["label1", "label2"]
        df[label1] = ["a", "b", "c"]
        df[label2] = ["1", "2", "3"]
        get_label_list(df, label_col) -> ["a\n1", "b\n2", "c\n3"]
    
    Args:
        df (pd.DataFrame): Dataframe
        label_col (str, list): Column name or list of column names
    
    Returns:
        list: List of labels
    """
    # Get possible labels
    if label_col:
        
        # Single label
        if isinstance(label_col, str):
            label_list = df[label_col].to_list()

        # Multiple labels
        if isinstance(label_col, list):
            
            # Make a list of lists
            ll = []
            for label in label_col:
                ll.append(list(map(str, df[label].to_list()))) # Map all values to strings
            
            # Combine the labels to a single string
            label_list = []
            for t in zip(*ll):
                label_list.append("\n".join(t))
                
        return label_list
    else:
        return None


class DataFrameVisualizer():
    """Visualization class for displaying images in a pandas dataframe. 
    """
    def __init__(self,
                 img_col,
                 parse_function=None,
                 img_size=64,
                 nrow=8):
        if not parse_function:
            self.parse_function = lambda x: x
        else:
            self.parse_function = parse_function
        
        self.img_col = img_col
        self.img_size = img_size
        self.nrow = nrow
        
    def __call__(self, 
                 df,
                 label_col=None,
                 **kwargs):
        
        # Set possible keyword arguments
        img_col = kwargs["img_col"] if "img_col" in kwargs.keys() else self.img_col
        img_size = kwargs["img_size"] if "img_size" in kwargs.keys() else self.img_size
        nrow = kwargs["nrow"] if "nrow" in kwargs.keys() else self.nrow
            
        # Parse filepaths
        fpath_list = df[img_col].apply(self.parse_function)
        
        label_list = get_label_list(df, label_col)
                    
        # Show files
        show_files(fpath_list, 
                   label_list if label_col else None,
                   nrow=nrow,
                   img_size=img_size)
        
    def __str__(self):
        s = (f"DataFrameVisualizer\n"
             f"Image column: {self.img_col}"
             )
        return s
    def __repr__(self):
        return self.__str__()
        
    def set_imgcol(self, img_col):
        """Changes the image column
        """
        c = copy.copy(self)
        c.img_col = img_col
        return c