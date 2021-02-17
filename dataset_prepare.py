import os
import shutil
import random 
import argparse
from tqdm import tqdm

random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--raw_dataset_pth', type=str, default='./raw-890', help='Path of raw dataset [default: ./raw-890]')
parser.add_argument('--ref_dataset_pth', type=str, default='./reference-890', help='Path of reference dataset [default: ././reference-890]')
parser.add_argument('--dataset_pth', type=str, default='./data', help='Path of reference dataset [default: ././reference-890]')

args = parser.parse_args()

RAW_PATH = args.raw_dataset_pth
REF_PATH = args.ref_dataset_pth
DATASET = args.dataset_pth

if not os.path.exists(DATASET):
    os.mkdir(DATASET)

raw_files = sorted(os.listdir(RAW_PATH))
ref_files = sorted(os.listdir(REF_PATH))

if len(raw_files) != len(ref_files):
    raise ValueError(f'Len raw images: {len(raw_files)}, does not match  reference images: {len(ref_files)}')
if raw_files != ref_files:
    raise ValueError(f'Raw files does not match reference files')

random.shuffle(raw_files)
train_files = raw_files[:int(0.8*len(raw_files))]
val_files = [filename for filename in raw_files if filename not in train_files]

train_dir = os.path.join(DATASET, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

for filename in tqdm(train_files):
    raw_file_pth = os.path.join(train_dir, 'raw')
    if not os.path.exists(raw_file_pth):
        os.mkdir(raw_file_pth)
    ref_file_pth = os.path.join(train_dir, 'ref')
    if not os.path.exists(ref_file_pth):
        os.mkdir(ref_file_pth)
    shutil.copy(os.path.join(RAW_PATH, filename), raw_file_pth)
    shutil.copy(os.path.join(REF_PATH, filename), ref_file_pth)

val_dir = os.path.join(DATASET, 'val')    
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

for filename in tqdm(val_files):
    raw_file_pth = os.path.join(val_dir, 'raw')
    if not os.path.exists(raw_file_pth):
        os.mkdir(raw_file_pth)
    ref_file_pth = os.path.join(val_dir, 'ref')
    if not os.path.exists(ref_file_pth):
        os.mkdir(ref_file_pth)
    shutil.copy(os.path.join(RAW_PATH, filename), raw_file_pth)
    shutil.copy(os.path.join(REF_PATH, filename), ref_file_pth)


