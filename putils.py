import os
import time
import torch
import random
import numpy as np
import config as cf
import torch.nn as nn
import nibabel as nib
import torch.optim as optim
import torch.nn.functional as F
from torch import amp
from dataset import BlocksDataset
from dataset import SubjectDataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from skimage.metrics import structural_similarity as ssim_fn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_block(x, size):
    min, max = np.min(x), np.max(x)
    len = max - min + 1
    num = int(np.ceil(len / size)) + 1
    start = min
    end = max - size + 1
    block = np.round(
        np.linspace(start, end, num)
    ).astype(int)

    return block

def get_block_index(mask, size=64):
    _, d, h, w = np.nonzero(mask)

    d_block = get_block(d, size)
    h_block = get_block(h, size)
    w_block = get_block(w, size)

    ind_block = []
    for d in d_block:
        for h in h_block:
            for w in w_block:
                ind_block.append([
                    d, d + size - 1,
                    h, h + size - 1,
                    w, w + size - 1
                ])

    return np.array(ind_block, dtype=int)


def get_blocks(image, index, expand=False):
    image = image[0]     # d, h, w

    d = index[0, 1] - index[0, 0] + 1
    h = index[0, 3] - index[0, 2] + 1
    w = index[0, 5] - index[0, 4] + 1

    n = index.shape[0]
    if (expand):
        blocks = np.zeros(
            (n, 1, d, h, w),
            dtype=image.dtype
        )
    else: 
        blocks = np.zeros(
            (n, d, h, w),
            dtype=image.dtype
        )

    for i in range(n):
        inds_this = index[i, :]

        if expand:
            blocks[i, 0, :, :, :] = image[
                inds_this[0]:inds_this[1] + 1,
                inds_this[2]:inds_this[3] + 1,
                inds_this[4]:inds_this[5] + 1
            ]
        else:
            blocks[i, :, :, :] = image[
                inds_this[0]:inds_this[1] + 1,
                inds_this[2]:inds_this[3] + 1,
                inds_this[4]:inds_this[5] + 1
            ]

    return blocks


def assemble_blocks(blocks, index, mask):
    n, c, _, _, _ = blocks.shape
    _, d, h, w = mask.shape

    image = np.zeros(
        (c, d, h, w),
        dtype=blocks.dtype
    )

    for i in range(n):
        d0, d1, h0, h1, w0, w1 = index[i]

        image[
            :,
            d0:d1 + 1,
            h0:h1 + 1,
            w0:w1 + 1
        ] = blocks[i]

    return image


def extract_blocks(x_patch, y_patch, x_files, y_files, block_size):
    idx = 0
    for i in range(len(x_files)):
        t1w_img = x_files[i].numpy()
        mask_img = y_files[0][i].numpy()
        block_index = get_block_index(mask_img, size=block_size)
        t1w_norm    = zscore_normalize(t1w_img, mask_img)
        t1w_blocks = get_blocks(t1w_norm, block_index)

        n = t1w_blocks.shape[0]
        x_patch[idx:idx+n, 0] = t1w_blocks
        for j in range(len(y_files)-1):
            y_blocks = get_blocks(y_files[j+1][i].numpy(), block_index)
            y_patch[idx:idx+n, j] = y_blocks

        idx += n


def zscore_normalize(x, mask):
    """
    x:      (1, d, h, w)
    mask:   (1, d, h, w)
    return: (1, d, h, w)
    """
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    assert x.shape == mask.shape, (
        f"x shape {x.shape} != mask shape {mask.shape}"
    )

    valid = (mask > 0) & np.isfinite(x)
    values = x[valid]
    mean = values.mean()
    std = values.std()

    std = max(float(std), 1e-6)

    x_norm = (x - mean) / std
    x_norm = np.where(valid, x_norm, 0.0)

    return x_norm.astype(np.float32)


def load_image(root_dir, subject_dir, file_name, expend=True, rescaling=False):
    full_path = os.path.join(root_dir,subject_dir, file_name)
    nii_file = nib.load(full_path)

    img_data = nii_file.get_fdata(dtype=np.float32)  # (x,y,z)
    if (rescaling):
        img_data = img_data / 3 * 1e3

    if expend :
        return np.expand_dims(img_data, axis=0) # (N, X,Y,Z)
    else : 
        return img_data                         # (X,Y,Z)


def load_data(source_dir):
    all_data_x = []
    all_data_y = []
    others = len(cf.FILE_NAMES) - 2
    count = 0

    for subject_dir in sorted(d for d in os.listdir(source_dir)
                        if os.path.isdir(os.path.join(source_dir, d))):
        count = count + 1
        dti_imgs = []

        t1w_img  = load_image(source_dir,subject_dir, cf.FILE_NAMES[0])
        mask_img = load_image(source_dir,subject_dir, cf.FILE_NAMES[1])
        dti_imgs.append(mask_img)

        for i in range(others):
            dti_imgs.append(load_image(source_dir,subject_dir, cf.FILE_NAMES[i+2], rescaling=cf.RESCALES[i+2]))

        all_data_x.append(t1w_img)
        all_data_y.append(dti_imgs)

    return all_data_x, all_data_y


def compute_ssim_mean(pred, true):
    ssim_vals = []
    for i in range(true.shape[2]):  # Z-dim
        dr = true[:, :, i].max() - true[:, :, i].min()

        if dr == 0:
            dr = 1.0

        ssim_val = ssim_fn(pred[:, :, i], true[:, :, i], data_range=dr)
        ssim_vals.append(ssim_val)

    return np.mean(ssim_vals)

def make_subject_loaders(
    x_files, y_files,
    batch_size=10, num_workers=0, augment=True
):
    ds = SubjectDataset(x_files, y_files)
    data_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    return data_loader    

def make_block_loaders(
    in_data: np.ndarray, out_datan: np.ndarray,
    batch_size: int=10, num_workers: int=0, augment: bool=True
):
    ds = BlocksDataset(in_data, out_datan, augment=augment)
    data_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    return data_loader


def fa_ssim_loss_3d(x, y, C1=0.01**2, C2=0.03**2):
    C1=0.01**2
    C2=0.03**2
    x   = torch.clamp(x, 0, 1)
    y   = torch.clamp(y, 0, 1)

    mu_x = F.avg_pool3d(x, 3, 1, 1)
    mu_y = F.avg_pool3d(y, 3, 1, 1)

    sigma_x = F.avg_pool3d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool3d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )

    return torch.clamp(ssim.mean(), 0, 1)


def compute_loss(pred_fa, pred_dti, gts, fa_ssim=0.4, two_head=False):
    if (two_head):
        loss_fa = F.l1_loss(pred_fa, gts[:, 0:1])
        loss_md = F.l1_loss(pred_dti[:, 0:1], gts[:, 1:2])
        loss_ad = F.l1_loss(pred_dti[:, 1:2], gts[:, 2:3])
        loss_rd = F.l1_loss(pred_dti[:, 2:3], gts[:, 3:4])

        loss = (loss_fa + loss_md + loss_ad + loss_rd) / 4
    else:
        loss = F.l1_loss(pred_dti, gts)

    if (fa_ssim > 0):
        # only add ssim for the first one (FA)
        if (two_head):
            loss += fa_ssim * (1 - fa_ssim_loss_3d(pred_fa, gts[:, 0:1, :, :, :]))
        else:
            loss += fa_ssim * (1 - fa_ssim_loss_3d(pred_dti[:, 0:1, :, :, :], gts[:, 0:1, :, :, :]))

    return loss


def compute_ssim(pred, gt, mask):
    if mask.ndim == 4:
        mask = mask[0]
    coords = np.where(mask > 0)

    xmin, xmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    zmin, zmax = coords[2].min(), coords[2].max()

    pred_crop = pred[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    gt_crop   = gt[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    mask_crop = mask[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

    ssim_list = []

    for i in range(pred_crop.shape[2]):
        gt_2d = gt_crop[:, :, i]
        pred_2d = pred_crop[:, :, i]
        mask_2d = mask_crop[:, :, i]

        if mask_2d.sum() == 0:
            continue

        val = ssim_fn(
            gt_2d,
            pred_2d,
            data_range=1.0
        )

        ssim_list.append(val)

    if len(ssim_list) == 0:
        return np.nan

    return np.mean(ssim_list)


def compute_metrics(pred, gt, mask=None):
    if mask is not None:
        if mask.ndim == 4:
            mask = mask[0]
        pred = pred[mask > 0]
        gt = gt[mask > 0]
    else:
        pred = pred.flatten()
        gt = gt.flatten()

    # MAE
    mae = np.mean(np.abs(pred - gt))

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # Pearson r
    r = np.corrcoef(pred, gt)[0, 1]

    return mae, rmse, r


def compute_result(results, pred, gt, i, name, mask=None):

    # ===== metrics =====
    ssim = compute_ssim(pred, gt, mask)
    mae, rmse, r = compute_metrics(pred, gt, mask)

    print(f" {name} -  MAE: {mae:.6f}   RMSE: {rmse:.6f}   r: {r:.6f}   SSIM: {ssim:.6f}")

    results[i][0].append(mae)
    results[i][1].append(rmse)
    results[i][2].append(r)
    results[i][3].append(ssim)


def compute_dti(L1, L2, L3):
    FA = np.sqrt(
        (
            (L1 - L2)**2 +
            (L2 - L3)**2 +
            (L3 - L1)**2
        )
        /
        ( 2 * (L1**2 + L2**2 + L3**2) + 1e-12 )
    )
    MD = (L1+L2+L3)/3
    AD = L1
    RD = (L2+L3)/2

    return FA, MD, AD, RD


def train(model, best_model_file, train_loader, val_loader, block_size=64, fa_ssim=0.4):
    two_head=model.two_head

    x_patch = np.empty(
        (cf.BATCH_FILE*cf.BLOCK_NUM, 1, block_size, block_size, block_size),
        dtype=np.float32
    )
    y_patch = np.empty(
        (cf.BATCH_FILE*cf.BLOCK_NUM, len(cf.FILE_NAMES)-2, block_size, block_size, block_size),
        dtype=np.float32
    )

    # optimizer & scaler
    opt = optim.Adam(model.parameters(), lr=3e-5)
    scaler = amp.GradScaler()

    num_epochs = cf.EPOCHS
    warmup_epochs = cf.WARMUP_EPOCHS
    scheduler = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=0.01, total_iters=warmup_epochs),
                CosineAnnealingLR(opt, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
            ],
            milestones=[warmup_epochs]
    )

    # --- Training config ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()
    begin_time = time.time()
    train_losses = []
    val_losses = []
    best_epoch = 0

    # --- Training loop ---
    for epoch in range(num_epochs):
        # ------ TRAIN --------
        start_time = time.time()
        model.train()
        print('Epoch:', epoch+1)
        running_train_loss = 0.0
        file_num = 0
        for x, y in train_loader:
            file_num += len(x)
            extract_blocks(x_patch, y_patch, x, y, block_size)
            block_loader = make_block_loaders(x_patch, y_patch, batch_size=cf.BATCH_SIZE, num_workers=0, augment=True)
            block_len = len(block_loader)

            for xb, yb in block_loader:
                xb, yb = xb.to(cf.DEVICE), yb.to(cf.DEVICE)
                opt.zero_grad()
        
                # === forward with AMP ===
                with amp.autocast("cuda"):  
                    if (two_head):
                        pred_fa, pred = model(xb)
                    else:
                        pred = model(xb)
                        pred_fa = None

                loss = compute_loss(pred_fa, pred, yb, fa_ssim=fa_ssim, two_head=two_head)

                # Backward
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                running_train_loss += loss.item() * xb.size(0)

        train_loss = running_train_loss / (len(train_loader.dataset) * block_len)

        # --- VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                extract_blocks(x_patch, y_patch, x, y, block_size)
                block_loader = make_block_loaders(x_patch, y_patch, batch_size=cf.BATCH_SIZE, num_workers=0, augment=False)
                block_len = len(block_loader)

                for xb, yb in block_loader:
                    xb, yb = xb.to(cf.DEVICE), yb.to(cf.DEVICE)
                    if (two_head):
                        pred_fa, pred = model(xb)
                    else:
                        pred = model(xb)
                        pred_fa = None

                    loss = compute_loss(pred_fa, pred, yb, fa_ssim=fa_ssim, two_head=two_head)
                    running_val_loss += loss.item() * xb.size(0)

        val_loss = running_val_loss / (len(val_loader.dataset) * block_len)

        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        print(f" Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f},  Val Loss = {val_loss:.6f},  Difference = {(val_loss-best_val_loss):.6f},  LR={current_lr:.6e},  Time = {(time.time()-start_time):.2f},  Total = {(time.time()-begin_time):.2f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # --- CHECKPOINT ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_file)
            print(f"  Improved val loss. Model saved (best_val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

    print(f"Training complete. Best validation loss = {best_val_loss:.6f}，  Best Epcho = {best_epoch},  Time = {time.time()-start_time}")

    return train_losses, val_losses


def test(source_dir, model, block_size=64, roi_mask_file=None):
    # --- Evalute mode by : MAE, RMSE, r, SSIM ---
    two_head=model.two_head
    count = 0
    pred_num = 4
    total = 4
    dti_names = []
    if cf.DTI_FILES is not None:
        pred_num = 3
        total = 7
        dti_names = ["FA", "MD", "AD", "RD"]

    x_patch = np.empty(
        (9, 1, block_size, block_size, block_size),
        dtype=np.float32
    )

    results = []
    for i in range(total):
        result = []
        results.append(result)
        for k in range(6):
            result.append([])

    roi_mask = None
    if (roi_mask_file is not None):
        roi_mask_nii = nib.load(roi_mask_file)
        roi_mask = roi_mask_nii.get_fdata(dtype=np.float32)

    for subject_dir in sorted(d for d in os.listdir(source_dir)
                        if os.path.isdir(os.path.join(source_dir, d))):
        count = count + 1
        print(f'Processing subject {count} : {subject_dir}')

        t1w_img     = load_image(source_dir,subject_dir, cf.FILE_NAMES[0])
        mask_img    = load_image(source_dir,subject_dir, cf.FILE_NAMES[1])
        t1w_img     = zscore_normalize(t1w_img, mask_img)
        block_index = get_block_index(mask_img, size=block_size)
        t1w_blocks  = get_blocks(t1w_img, block_index, expand=True)

        # Run network inference
        net_in = torch.from_numpy(t1w_blocks)

        img_gt = []
        dti_gt = []
        for i in range(pred_num):
            img = load_image(source_dir,subject_dir, cf.FILE_NAMES[i+2], rescaling=cf.RESCALES[i+2])
            img_gt.append(img)

        # Eigenvector File Names
        if pred_num == 3:
            for i in range(len(cf.DTI_FILES)):
                rescaling = i > 0
                #(X,Y,Z)
                img = load_image(source_dir,subject_dir, cf.DTI_FILES[i], expend=False, rescaling=rescaling)
                dti_gt.append(img)

        with torch.no_grad():
            preds = []
            for k in range(int(net_in.shape[0] / x_patch.shape[0])):
                x_patch = net_in[k * 9:(k + 1) * 9]

                if two_head:
                    pred_fa, pred = model(x_patch.to(device=cf.DEVICE, dtype=torch.float))
                    pred = torch.cat([pred_fa, pred], dim=1)
                else:
                    pred = model(x_patch.to(device=cf.DEVICE, dtype=torch.float))
                    pred_fa = None

                preds.append(pred)

        y_pred = torch.cat(preds, dim=0)
        y_pred = y_pred.cpu().numpy()
        y_pred_wb = assemble_blocks(y_pred, block_index, mask_img)

        # Pred results
        if (roi_mask is not None):
            mask = roi_mask
        else:
            mask = mask_img > 0

        for i in range(pred_num):
            pred = y_pred_wb[i,:,:,:]  
            gt = img_gt[i][0,:,:,:]  

            compute_result(results, pred, gt, i, cf.FILE_NAMES[i+2], mask)

        # compute FA/MD/AD/RD and results
        if len(dti_gt) > 0:
            pred_l1 = y_pred_wb[0,:,:,:]
            pred_l2 = y_pred_wb[1,:,:,:]
            pred_l3 = y_pred_wb[2,:,:,:]
            pred_fa, pred_md, pred_ad, pred_rd = compute_dti(pred_l1, pred_l2, pred_l3)
            dti_pred = []
            dti_pred.append(pred_fa)
            dti_pred.append(pred_md)
            dti_pred.append(pred_ad)
            dti_pred.append(pred_rd)

            for k in range(len(dti_pred)):
                pred = dti_pred[k]  
                gt = dti_gt[k]

                compute_result(results, pred, gt, pred_num+k, dti_names[k], mask=mask)

    # ===== FINAL =====
    print("\n===== FINAL RESULTS =====")

    for i in range(pred_num) : 
        print(f"{cf.FILE_NAMES[i+2]} :  MAE = {np.mean(results[i][0]):.6f},  RMSE = {np.mean(results[i][1]):.6f},  r = {np.mean(results[i][2]):.6f},  SSIM = {np.mean(results[i][3]):.6f}")

    if pred_num == 3:
        for i in range(len(dti_names)) : 
            print(f"{dti_names[i]} :  MAE = {np.mean(results[i+pred_num][0]):.6f},  RMSE = {np.mean(results[i+pred_num][1]):.6f},  r = {np.mean(results[i+pred_num][2]):.6f},  SSIM = {np.mean(results[i+pred_num][3]):.6f}")
