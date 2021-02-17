import numpy as np 
import math
import random
import os
import sys
import json
import torch
import argparse
import datetime
from tqdm import tqdm

from uieb_dataset import *
from models.decomp import *
from data_utils import *

torch.manual_seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--raw_dataset_pth', type=str, default='./raw-890', help='Path of raw dataset [default: ./raw-890]')
parser.add_argument('--ref_dataset_pth', type=str, default='./reference-890', help='Path of reference dataset [default: ././reference-890]')
parser.add_argument('--weights_path', type=str, default=None, help='Path to pretrained model [default: None]')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs [default: 200]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--exp_dir', type=str, default=None, help='Experiment dir [default: log]')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate for lr decay [default: 1e-4]')
args = parser.parse_args()

RAW_PATH = args.raw_dataset_pth
REF_PATH = args.ref_dataset_pth
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
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)
log_dir = os.path.join(EXP_DIR, 'logs')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

""" Data loading """
dataset = UiebDataset(RAW_PATH, REF_PATH)
lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_ds, valid_ds = random_split(init_dataset, lengths)
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=BATCH_SIZE)

""" Model loading """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
decomp = Decompose(num_layers=5)
decomp.to(device)

""" Define optimizer """
optimizer = torch.optim.Adam(
                pointnet.parameters(),
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

""" Training """
for epoch in range(start_epoch, EPOCHS): 
    print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, EPOCHS))
    running_loss = 0.0
    correct = total = 0
    # train
    for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        raw_img, ref_img = dataset
        raw_img.to(device), ref_img.to(device)
        
        rand_mode = random.randint(0, 7)
        raw_img = torch.Tensor(data_augmentation(raw_img), rand_mode)
        ref_img = torch.Tensor(data_augmentation(ref_img), rand_mode)

        optimizer.zero_grad()
        decomp.train()

        R_low, I_low = decomp(raw_img)
        R_high, I_high = decomp(ref_img)
        loss = pointnetloss(outputs, labels, m64x64)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).cpu().sum().item()

        loss.backward()
        optimizer.step()
        global_step +=1

    train_acc = 100. * correct / total
    print('Train accuracy: %f' % train_acc)

    pointnet.eval()
    correct = total = 0
    # validation
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            outputs, __ = pointnet(inputs.transpose(1,2))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = 100. * correct / total
        print('Valid accuracy: %d %%' % val_acc)
        print('Best valid accuracy: %d %%' % best_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            print('Saving model...')
            savepth = os.path.join(checkpoints_dir, f'/saved_epoch_{epoch+1}.pth')
            print(f'Model saved at {savepth}')
            state = {
                    'epoch': epoch,
                    'acc': best_acc,
                    'model_state_dict': pointnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }
            torch.save(state, savepth)
            global_epoch += 1
    scheduler.step()