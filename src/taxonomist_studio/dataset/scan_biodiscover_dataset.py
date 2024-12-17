"""Reads image files from a image dataset produced by the BioDiscover imaging machine.
"""

import os
from pathlib import Path

import pandas as pd

from taxonomist_studio import tools


def add_args(parser):
    group = parser.add_argument_group("scan")
    group.add_argument('--data_folder')
    group.add_argument('-o', '--out', default=None, required=False)
    group.add_argument('--out_folder', default='.', required=False)
    group.add_argument('--category_file', default=None, required=False)
    group.add_argument('--species_level', action="store_true")
    group.add_argument('--size', action="store_true")
    return group

def scan(args):
    
    path = Path(args.data_folder)

    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    # Reads possible higher level category folders
    if args.category_file:
        category_file = Path(args.category_file)
        category_name = category_file.stem
        with open(category_file) as f:
            categories = [x.strip() for x in f.readlines()]
    elif args.species_level:
        category_name = "species_name"
        categories = os.listdir(path)
    else:
        categories = [path.stem]
        category_name = "folder"
        path = (path / '..').resolve()

    cat_list = []
    ind_list = []
    sam_list = []
    img_list = []
    h_list = []
    w_list = []

    # Go through categories
    for cat in categories:
        cat_folder = Path(path, cat)
        individuals = os.listdir(cat_folder)
        print("individuals:", len(individuals))
        for ind in individuals:
            individual_folder = Path(cat_folder, ind)
            samples = os.listdir(individual_folder)
            if "Calibration" in samples:
                samples.remove('Calibration')
            else:
                print(f"No calibration folder in {cat}, {ind}")
            print(f"\t{ind} samples:", len(samples))
            for sam in samples:
                sample_folder = Path(individual_folder, sam)
                imgs = [x.name for x in sample_folder.glob('*.PNG')]
                for x in imgs:
                    if not str(x).endswith('PNG'):
                        print(f"{x} not a png")
                print("\t\timages:", len(imgs))
                for img in imgs:
                    cat_list.append(cat)
                    ind_list.append(ind)
                    sam_list.append(sam)
                    img_list.append(img)
                    if args.size:
                        size = tools.get_image_size(Path(sample_folder, img))
                        h_list.append(size[0])
                        w_list.append(size[1])

    df = pd.DataFrame({category_name:cat_list, 'individual':ind_list, 'sample': sam_list, 'image':img_list})

    if args.size:
        df = df.assign(height=h_list,
                        width=w_list)

    if not args.out:
        args.out = "imagescan.csv"
    out_fname = out_folder / f"{args.out}"
    print(out_fname)

    df.to_csv(out_fname, index=False)