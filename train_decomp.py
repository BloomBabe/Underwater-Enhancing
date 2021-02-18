import numpy as np 
import math
import random
import os
import sys
import json
import torch
import argparse
import datetime
from skimage import io
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from uieb_dataset import *
from models.decomp import *
from data_utils import *

torch.manual_seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_pth', type=str, default='./data', help='Path of reference dataset [default: ././reference-890]')
parser.add_argument('--image_shape', type=tuple, default=(256, 256), help='Shape of imput images')
parser.add_argument('--weights_path', type=str, default=None, help='Path to pretrained model [default: None]')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs [default: 200]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--exp_dir', type=str, default=None, help='Experiment dir [default: log]')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate for lr decay [default: 1e-4]')
args = parser.parse_args()

DATASET = args.dataset_pth
INPUT_SHAPE = args.image_shape
WEIGHTS_PTH = args.weights_path
EXP_DIR = args.exp_dir
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
LR_RATE = args.learning_rate
DECAY_RATE = args.decay_rate

""" Create experiment directory """
timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
if EXP_DIR is None:
    EXP_DIR = os.path.join('experiments', 'decomposition', timestr)
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)
checkpoints_dir = os.path.join(EXP_DIR, 'checkpoints')
print(checkpoints_dir)
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)
log_dir = os.path.join(EXP_DIR, 'logs')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

""" Data loading """
transform = transforms.Compose([ToTensor(), Resize((256, 256)), 
                                Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])
train_ds = UiebDataset(DATASET, mode='train', transform=transform)
valid_ds = UiebDataset(DATASET, mode='val', transform=transform)
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=BATCH_SIZE)

""" Model loading """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
decomp = Decompose(num_layers=5)
decomp = decomp.float()
decomp.to(device)

""" Define optimizer """
optimizer = torch.optim.Adam(
                decomp.parameters(),
                lr=LR_RATE,
                weight_decay=DECAY_RATE
                )

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

""" Load model weights """
if WEIGHTS_PTH is not None:
    checkpoint = torch.load(WEIGHTS_PTH)
    start_epoch = checkpoint['epoch']
    decomp.load_state_dict(checkpoint['model_state_dict'])
    print('Use pretrained model')
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
else:
    start_epoch = 0

global_epoch = 0
global_step = 0

best_loss = 100.
""" Training """
for epoch in range(start_epoch, EPOCHS): 
    print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, EPOCHS))
    mean_loss = 0.
    # train
    for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        raw_img, ref_img = data['raw_image'].to(device), data['ref_image'].to(device)
        img = raw_img[1,:,:,:].cpu().detach().numpy()
        io.imsave('/content/file.png', img.transpose((1, 2, 0)))
        optimizer.zero_grad()
        decomp.train()
        R_low, I_low = decomp(raw_img)
        R_high, I_high = decomp(ref_img)
        loss = deocmp_loss(ref_img, raw_img, R_high, I_high, R_low, I_low)
        mean_loss += float(loss)
        loss.backward()
        optimizer.step()
        global_step +=1
    print(f'Mean loss: {mean_loss/(batch_id+1):.4f}')

    decomp.eval()
    # validation
    with torch.no_grad():
        val_loss = 0.
        for i, data in enumerate(valid_loader):
            raw_img, ref_img = data['raw_image'].to(device), data['ref_image'].to(device)
            R_low, I_low = decomp(raw_img)
            R_high, I_high = decomp(ref_img)
            loss = deocmp_loss(ref_img, raw_img, R_high, I_high, R_low, I_low)
            val_loss += float(loss)
        val_loss = val_loss/(i+1)
        print(f'Valid mean loss: {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            print('Saving model...')
            savepth = os.path.join(checkpoints_dir, f'saved_epoch_{epoch+1}.pth')
            print(f'Model saved at {savepth}')
            state = {
                    'epoch': epoch,
                    'loss': best_loss,
                    'model_state_dict': decomp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }
            torch.save(state, savepth)
            global_epoch += 1
    scheduler.step()