"""
HaMeR ViT-B Training v2
Improvements over v1:
- 2D reprojection loss (solves the localization problem)
- Warmup LR + Cosine Annealing
- Early stopping
- Weighted losses as in the original paper
"""

import os
import sys
import time
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─── Paths ───────────────────────────────────────────────
BASE_DIR     = '/scratch.hpc/paolo.fusco2/hamer'
FREIHAND_DIR = '/scratch.hpc/paolo.fusco2/FreiHAND'
NPZ_PATH     = f'{BASE_DIR}/hamer_training_data/freihand_train_v2.npz'
CONFIG_PATH  = f'{BASE_DIR}/_DATA/hamer_ckpts_vitb/model_config.yaml'
MAE_WEIGHTS  = f'{BASE_DIR}/_DATA/vitpose_ckpts/vitpose_base/wholebody_base.pth'
CKPT_DIR     = f'{BASE_DIR}/checkpoints_vitb_v2'

# ─── Iperparametri ───────────────────────────────────────
NUM_EPOCHS    = 30
BATCH_SIZE    = 32
LR            = 5e-5        # LR iniziale più conservativo
LR_MAX        = 1e-4        # LR massimo dopo warmup
WARMUP_EPOCHS = 3           # epoch di warmup
LOG_EVERY     = 50

# Pesi loss (come nel paper originale)
W_KP3D  = 0.05
W_KP2D  = 0.01
W_BETA  = 0.0005

# Early stopping
ES_PATIENCE = 5   # ferma se val loss non migliora per N epoch
ES_MIN_DELTA = 1e-4

os.makedirs(CKPT_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

# ─── Setup ───────────────────────────────────────────────
print("=" * 60)
print("HaMeR ViT-B Training v2")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── Config ──────────────────────────────────────────────
from hamer.configs import get_config
from hamer.models import HAMER

model_cfg = get_config(CONFIG_PATH, update_cachedir=True)
model_cfg.defrost()
if 'BBOX_SHAPE' not in model_cfg.MODEL:
    model_cfg.MODEL.BBOX_SHAPE = [192, 256]
if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
    model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
model_cfg.freeze()

print(f"Backbone: {model_cfg.MODEL.BACKBONE.TYPE}")

# ─── Modello ─────────────────────────────────────────────
model = HAMER(cfg=model_cfg)

mae_weights = torch.load(MAE_WEIGHTS, map_location='cpu')
if 'state_dict' in mae_weights:
    mae_weights = mae_weights['state_dict']

if 'pos_embed' in mae_weights:
    pos_embed_old = mae_weights['pos_embed']
    cls_token   = pos_embed_old[:, :1, :]
    patch_embed = pos_embed_old[:, 1:, :]
    patch_embed = patch_embed.reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(patch_embed, size=(16, 12), mode='bicubic', align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, 192, 768)
    mae_weights['pos_embed'] = torch.cat([cls_token, patch_embed], dim=1)
    print(f"pos_embed interpolato: {mae_weights['pos_embed'].shape}")

missing, unexpected = model.backbone.load_state_dict(mae_weights, strict=False)
print(f"Pesi MAE caricati — Missing: {len(missing)}, Unexpected: {len(unexpected)}")

model = model.to(device)
print(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")

# ─── Dataset ─────────────────────────────────────────────
from hamer.datasets.image_dataset import ImageDataset

dataset = ImageDataset(
    cfg          = model_cfg,
    dataset_file = NPZ_PATH,
    img_dir      = FREIHAND_DIR,
    train        = True,
)
print(f"Dataset caricato: {len(dataset)} campioni")

n_val   = int(0.1 * len(dataset))
n_train = len(dataset) - n_val
train_set, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Training: {n_train} | Validation: {n_val} | Batch/epoch: {len(train_loader)}")

# ─── Optimizer e scheduler con warmup ────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# CosineAnnealing dopo warmup
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

def get_lr_with_warmup(epoch):
    """Warmup lineare per le prime WARMUP_EPOCHS epoch"""
    if epoch < WARMUP_EPOCHS:
        return LR + (LR_MAX - LR) * epoch / WARMUP_EPOCHS
    return None  # usa scheduler

# ─── Checkpoint ──────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, scheduler, loss, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss':                 loss,
    }, path)
    print(f"Checkpoint salvato: {path}")

def load_checkpoint_if_exists(model, optimizer, scheduler):
    ckpt_path = f'{CKPT_DIR}/latest.ckpt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Ripreso dal checkpoint — epoch {start_epoch + 1}")
        return start_epoch
    print("Nessun checkpoint trovato — training da zero")
    return 0

start_epoch = load_checkpoint_if_exists(model, optimizer, scheduler)

# ─── Early stopping ──────────────────────────────────────
best_val_loss = float('inf')
es_counter    = 0

# ─── Training loop ───────────────────────────────────────
for epoch in range(start_epoch, NUM_EPOCHS):

    # Warmup LR manuale
    warmup_lr = get_lr_with_warmup(epoch)
    if warmup_lr is not None:
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr
        current_lr = warmup_lr
    else:
        current_lr = scheduler.get_last_lr()[0]

    # TRAINING
    model.train()
    train_loss = 0.0
    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

        kp3d_gt  = batch['keypoints_3d'].to(device)   # (B, 21, 4)
        has_kp3d = kp3d_gt[:, :, 3]                    # (B, 21) confidence
        kp2d_gt  = batch['keypoints_2d'].to(device)    # (B, 21, 3)
        has_kp2d = kp2d_gt[:, :, 2]                    # (B, 21) confidence

        out = model(batch_device)

        # Loss keypoint 3D
        kp3d_pred = out['pred_keypoints_3d']            # (B, 21, 3)
        loss_kp3d = (((kp3d_pred - kp3d_gt[:,:,:3]) ** 2).sum(-1) * has_kp3d).mean()

        # Loss keypoint 2D — reprojection
        kp2d_pred = out['pred_keypoints_2d']            # (B, 21, 2)
        # Normalizza gt 2D in [-1, 1] rispetto all'image size
        img_size  = model_cfg.MODEL.IMAGE_SIZE
        kp2d_gt_norm = kp2d_gt[:, :, :2] / img_size * 2 - 1
        loss_kp2d = (((kp2d_pred - kp2d_gt_norm) ** 2).sum(-1) * has_kp2d).mean()

        # Loss betas
        has_beta  = batch['has_mano_params']['betas'].to(device)
        loss_beta = torch.tensor(0.0, device=device)
        if has_beta.sum() > 0:
            beta_pred = out['pred_mano_params']['betas']
            beta_gt   = batch['mano_params']['betas'].to(device)
            loss_beta = (((beta_pred - beta_gt) ** 2).sum(-1) * has_beta).mean()

        loss = W_KP3D * loss_kp3d + W_KP2D * loss_kp2d + W_BETA * loss_beta
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % LOG_EVERY == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                  f"Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.5f} | "
                  f"kp3d: {loss_kp3d.item():.4f} | "
                  f"kp2d: {loss_kp2d.item():.4f} | "
                  f"beta: {loss_beta.item():.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Elapsed: {elapsed:.0f}s")
            sys.stdout.flush()

    avg_train_loss = train_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
            kp3d_gt   = batch['keypoints_3d'].to(device)
            has_kp3d  = kp3d_gt[:, :, 3]
            kp2d_gt   = batch['keypoints_2d'].to(device)
            has_kp2d  = kp2d_gt[:, :, 2]
            out       = model(batch_device)
            kp3d_pred = out['pred_keypoints_3d']
            kp2d_pred = out['pred_keypoints_2d']
            kp2d_gt_norm = kp2d_gt[:, :, :2] / img_size * 2 - 1
            l3d = (((kp3d_pred - kp3d_gt[:,:,:3]) ** 2).sum(-1) * has_kp3d).mean()
            l2d = (((kp2d_pred - kp2d_gt_norm) ** 2).sum(-1) * has_kp2d).mean()
            val_loss += (W_KP3D * l3d + W_KP2D * l2d).item()

    avg_val_loss = val_loss / len(val_loader)

    # Scheduler step (solo dopo warmup)
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} completata")
    print(f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
    print(f"LR: {current_lr:.6f}")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()

    # Salva checkpoint
    save_checkpoint(epoch, model, optimizer, scheduler, avg_val_loss,
                    f'{CKPT_DIR}/epoch_{epoch+1:02d}.ckpt')
    save_checkpoint(epoch, model, optimizer, scheduler, avg_val_loss,
                    f'{CKPT_DIR}/latest.ckpt')

    # Tieni solo ultimi 3 checkpoint
    ckpts = sorted(glob.glob(f'{CKPT_DIR}/epoch_*.ckpt'))
    for old in ckpts[:-3]:
        os.remove(old)
        print(f"Rimosso: {old}")

    # Early stopping
    if avg_val_loss < best_val_loss - ES_MIN_DELTA:
        best_val_loss = avg_val_loss
        es_counter    = 0
        # Salva il miglior modello separatamente
        save_checkpoint(epoch, model, optimizer, scheduler, avg_val_loss,
                        f'{CKPT_DIR}/best.ckpt')
        print(f"Nuovo best val loss: {best_val_loss:.5f}")
    else:
        es_counter += 1
        print(f"Early stopping counter: {es_counter}/{ES_PATIENCE}")
        if es_counter >= ES_PATIENCE:
            print(f"Early stopping attivato a epoch {epoch+1}!")
            print(f"Best val loss: {best_val_loss:.5f}")
            break

print("\nTraining completato!")
print(f"Best val loss: {best_val_loss:.5f}")
