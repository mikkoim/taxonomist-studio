from pathlib import Path
import pandas as pd
from pathlib import Path
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import json
import numpy as np

def get_transform(resize):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform

def image_loader(fname, resize=224):
    tfm = get_transform(resize)
    img = Image.open(fname).convert("RGB")
    return tfm(img)

class ImageDataset(Dataset):
    def __init__(self, fnames):
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        return image_loader(self.fnames[idx]), str(self.fnames[idx].name)

def run_embedder(args):
    df = pd.read_csv(args.csv_path, sep=";", encoding="ISO-8859-1")
    out_folder = Path(args.out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    if args.head:
        df = df.sample(args.head)

    fpaths = df.apply(
        lambda x: Path(
            args.data_folder,
            *Path(x["Image Path"]).parts[-3:],
            x["Image File Name"],
        ),
        axis=1,
    )
    fpaths = [x for x in fpaths]

    print(f"Using model {args.timm_model}")
    print(f"Found {len(fpaths)} images")
    print(f"Batch size: {args.batch_size}")
    print(f"Total batches: {int(np.ceil(len(fpaths) /args.batch_size))}")
    dataloader = DataLoader(ImageDataset(fpaths),
                            batch_size=args.batch_size,
                            shuffle=False)

    m = timm.create_model(args.timm_model, pretrained=True, num_classes=0)
    frozen_m = torch.jit.optimize_for_inference(torch.jit.script(m.eval()))

    features = []
    fnames = []
    for b, fname_b in tqdm(dataloader):
        features.append(frozen_m(b).numpy().tolist())
        fnames.append(fname_b)

    features = np.concatenate(features).astype(np.float16) # float16 to save space
    fnames = np.concatenate(fnames)
    feature_df = pd.DataFrame(features, index=fnames)

    feature_df.columns = feature_df.columns.astype(str) # needed for parquet
    feature_df.to_parquet(out_folder / args.out_fname, compression="gzip", index=True)

    print(f"Saved to {out_folder / args.out_fname}")