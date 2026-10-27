import torch
import time
import putils as ut
import config as cf
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch import amp
from modules import TransUNet
from modules import UNet3D
from modules import CNN3D
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
torch.cuda.set_per_process_memory_fraction(0.90)

# --- Load Data ---

# training data
all_data_x, all_data_y = ut.load_data(cf.TRAIN_DIR)
train_loader = ut.make_subject_loaders(all_data_x, all_data_y, batch_size=cf.BATCH_SIZE)
print(f"All training data loaded: all_data_x = {len(all_data_x)},  all_data_y = {len(all_data_y)}")

# validation data
all_data_x, all_data_y = ut.load_data(cf.VALIDATION_DIR)
val_loader = ut.make_subject_loaders(all_data_x, all_data_y, batch_size=cf.BATCH_SIZE)
print(f"All validation data loaded: all_data_x = {len(all_data_x)},  all_data_y = {len(all_data_y)}")

# ---- Training ----
if cf.MODEL == "TransUNet":
    model = TransUNet(two_head=cf.TWO_HEAD).to(cf.DEVICE)
    best_model_file = cf.TRANSUNET_BEST_MODEL
elif cf.MODEL == "UNet":
    model = UNet3D().to(cf.DEVICE)
    best_model_file = cf.UNET_BEST_MODEL
else: 
    model = CNN3D().to(cf.DEVICE)
    best_model_file = cf.CNN_BEST_MODEL

ut.set_seed(cf.SEED)
print(f"Training: Model = {cf.MODEL},  Patch = {cf.BLOCK_SIZE},  Two Head = {cf.TWO_HEAD},  FA SSIM Loss = {cf.FA_SSIM},  best_model = {best_model_file}")
train_losses, val_losses = ut.train(model, best_model_file, train_loader, val_loader, block_size=cf.BLOCK_SIZE, fa_ssim=cf.FA_SSIM)
