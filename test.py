import os
import torch
import numpy as np
import torch.nn as nn
import config as cf
import putils as ut
from modules import TransUNet
from modules import UNet3D
from modules import CNN3D
torch.cuda.set_per_process_memory_fraction(0.90)


if cf.MODEL == "TransUNet":
    model = TransUNet(two_head=cf.TWO_HEAD).to(cf.DEVICE)
    best_model_file = cf.TRANSUNET_BEST_MODEL
elif cf.MODEL == "UNet":
    model = UNet3D().to(cf.DEVICE)
    best_model_file = cf.UNET_BEST_MODEL
else: 
    model = CNN3D().to(cf.DEVICE)
    best_model_file = cf.CNN_BEST_MODEL

state = torch.load(best_model_file, map_location="cpu")
model.load_state_dict(state, strict=False)

ut.set_seed(cf.SEED)
ut.test(cf.TEST_DIR, model, block_size=cf.BLOCK_SIZE, roi_mask_file=cf.ROI_MASK_FILE)
