"""
Aggiorna freihand_train.npz aggiungendo i keypoint 2D
calcolati dalla proiezione prospettica K @ xyz
"""
import json
import numpy as np

print("Caricamento dati FreiHAND...")
with open('/scratch.hpc/paolo.fusco2/FreiHAND/training_xyz.json') as f:
    xyz_all = json.load(f)
with open('/scratch.hpc/paolo.fusco2/FreiHAND/training_K.json') as f:
    K_all = json.load(f)

n_base  = len(xyz_all)   # 32560
n_aug   = 4
n_total = n_base * n_aug # 130240

print(f"Calcolando {n_total} keypoint 2D...")
kps_2d = []

for aug_idx in range(n_aug):
    for i in range(n_base):
        xyz = np.array(xyz_all[i], dtype=np.float32)  # (21, 3)
        K   = np.array(K_all[i],   dtype=np.float32)  # (3, 3)

        # Proiezione prospettica: pixel = K @ xyz / z
        proj = (K @ xyz.T)          # (3, 21)
        u    = proj[0] / proj[2]    # x pixel
        v    = proj[1] / proj[2]    # y pixel
        conf = np.ones(21, dtype=np.float32)

        kps_2d.append(np.stack([u, v, conf], axis=1))  # (21, 3)

    print(f"  augmentation {aug_idx+1}/4 completata")

kps_2d = np.array(kps_2d, dtype=np.float32)  # (130240, 21, 3)
print(f"keypoint 2D shape: {kps_2d.shape}")
print(f"range u: {kps_2d[:,:,0].min():.1f} / {kps_2d[:,:,0].max():.1f}")
print(f"range v: {kps_2d[:,:,1].min():.1f} / {kps_2d[:,:,1].max():.1f}")

# Carica NPZ esistente e aggiungi keypoint 2D
data = np.load('/scratch.hpc/paolo.fusco2/hamer/hamer_training_data/freihand_train.npz',
               allow_pickle=True)

out_path = '/scratch.hpc/paolo.fusco2/hamer/hamer_training_data/freihand_train_v2.npz'
np.savez(out_path,
    imgname           = data['imgname'],
    center            = data['center'],
    scale             = data['scale'],
    hand_pose         = data['hand_pose'],
    betas             = data['betas'],
    has_hand_pose     = data['has_hand_pose'],
    has_betas         = data['has_betas'],
    hand_keypoints_3d = data['hand_keypoints_3d'],
    hand_keypoints_2d = kps_2d,
    right             = data['right'],
)
print(f"NPZ v2 salvato: {out_path}")
