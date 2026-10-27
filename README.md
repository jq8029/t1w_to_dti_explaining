## Understanding Target-Dependent Prediction Difficulty in Voxelwise DTI Metric Prediction from Structural MRI

## Overview

This project investigates **target-dependent prediction difficulty in cross-modal synthesis of diffusion tensor imaging (DTI) metrics from T1-weighted structural MRI**.

We predict four DTI metrics: **fractional anisotropy (FA), mean diffusivity (MD), axial diffusivity (AD), and radial diffusivity (RD)**. The study examines whether differences in prediction performance are associated with target mathematical properties, data distribution, and model architecture.

To investigate these factors, we compare CNN, U-Net, and TransUNet architectures and evaluate the effects of spatial context, target representation, output-head configuration, and FA-specific structural similarity (SSIM) supervision. Regional prediction performance is also analyzed across four major white matter structures.

The goal is to understand **why some DTI metrics are more difficult to predict than others**, rather than simply maximizing overall reconstruction accuracy.


## Methods
- Architectures: CNN, U-Net, TransUNet
- Targets: FA, MD, AD, RD
- Patch sizes: 32³, 64³, 96³
- Target representations: Direct prediction and
  eigenvalue-based reconstruction
- Output configurations: Single-head and two-head
- Loss functions: L1 and FA-specific SSIM loss

## Dataset 

- **Dataset**: Human Connectome Project (HCP)
- **Subjects**: 160
- **Training set**: 90 subjects
- **Validation set**: 30 subjects
- **Test set**: 40 subjects
- **Input**: T1-weighted structural MRI
- **Targets**: FA, MD, AD, and RD
- **Spatial resolution**: 1.25 × 1.25 × 1.25 mm in MNI space


## Evaluation Metrics

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**
- **Pearson correlation coefficient (r)**
- **SSIM (Structural Similarity Index Measure)**


## Key Findings
- FA exhibits greater prediction difficulty than
  the other DTI metrics under the baseline setting.
- TransUNet improves performance compared with
  CNN and U-Net, but target-dependent differences persist.
- Eigenvalue-based reconstruction affects FA more
  strongly than MD, AD, and RD.
- FA-specific SSIM supervision improves FA metrics
  but may increase reconstruction errors for other targets.
- Prediction performance varies across white matter regions.

## Experimental Settings

Each experiment is conducted using **five random seeds**: `8, 42, 356, 2026, and 8096`.

Experiments follow a **one-factor-at-a-time (OFAT) design**, in which only one experimental variable is changed at a time in **config.py** while all other settings are kept fixed. This allows the effect of each factor on DTI prediction performance to be evaluated independently.

| ID | Experiment | BLOCK_SIZE | BLOCK_NUM | TWO_HEAD | FA_SSIM | Model |
|---|---|---|---|---|---|---|
| E0 | Baseline | **64** | **36** | **False** | **0.0** | **TransUNet** |
| E1 | Architecture comparison | 64 | 36 | False | 0.0 | **CNN** / **U-Net** |
| E2 | Patch size comparison | **32 / 96** | **150 / 27** | False | 0.0 | TransUNet |
| E3 | SSIM loss comparison | 64 | 36 | False | **0.2 / 0.4** | TransUNet |
| E4 | Output-head comparison | 64 | 36 |  **True** | 0.0 | TransUNet |
| E5 | Indirect prediction | 64 | 36 | False | 0.0 | TransUNet |

## Usage

- **PreProcess**  
./hcp_preprocess.sh

- **Training**  
python train.py

- **Test**  
python test.py

