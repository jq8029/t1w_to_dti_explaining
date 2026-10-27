import torch

# File Options

# DTI File Names
FILE_NAMES = ["T1_0_brain.nii.gz","T1_mask_0_brain.nii.gz", "T1_1_FA.nii.gz", "T1_2_MD.nii.gz", "T1_3_AD.nii.gz", "T1_4_RD.nii.gz"]
RESCALES   = [False, False, False, True, True, True]
DTI_FILES  = None

# ROI File Name
ROI_MASK_FILE = None   #ROI_CC_HCP125.nii.gz, ROI_CR_HCP125.nii.gz, ROI_IC_HCP125.nii.gz, ROI_SLF_HCP125.nii.gz

# Eigenvector File Names
#FILE_NAMES = ["T1_0_brain.nii.gz","T1_mask_0_brain.nii.gz", "T1_11_L1.nii.gz", "T1_12_L2.nii.gz", "T1_13_L3.nii.gz"]
#RESCALES   = [False, False, True, True, True]
#DTI_FILES  = ["T1_1_FA.nii.gz", "T1_2_MD.nii.gz", "T1_3_AD.nii.gz", "T1_4_RD.nii.gz"]

# Path Options
TRAIN_DIR      = "/prject/data/train"
VALIDATION_DIR = "/project/data/val"
TEST_DIR       = "/project/data/test"

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_FILE     = 10
BATCH_SIZE     = 10
EPOCHS         = 100
WARMUP_EPOCHS  = 10
TRANSUNET_BEST_MODEL = "transunet_best_model.pt"
UNET_BEST_MODEL      = "unet_best_model.pt"
CNN_BEST_MODEL       = "cnn_best_model.pt"

# Options
SEED           = 356          # 8, 42, 356, 2026, 8096
BLOCK_SIZE     = 64           # 32, 64, 96  
BLOCK_NUM      = 36           # 150 - 32, 36 - 64;  27 - 96 
TWO_HEAD       = False        # True, False
FA_SSIM        = 0.0          # 0.0, 0.2, 0.4
MODEL          = "TransUNet"  # TransUNet, UNet, CNN