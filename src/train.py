# ----------------------------------------------------------------------
# train.py – Final Stable Version (Dynamic Metrics)
# ----------------------------------------------------------------------
from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
from config import Config as cfg

import torch
import numpy as np
import os
from tqdm import tqdm
import sys

# AMP Import (supports all PyTorch versions)
try:
    from torch.amp import autocast, GradScaler
    NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    NEW_AMP = False


print("Initializing DeepCrack training...")
print(f"Checkpoints will be saved in: {cfg.saver_path}")

print("Loading dataset indices...")
train_pipeline = dataReadPip(is_train=True)
val_pipeline   = dataReadPip(is_train=False)

train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=True), preprocess=train_pipeline)
val_dataset   = loadedDataset(readIndex(cfg.test_data_path), preprocess=val_pipeline)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg.train_batch_size,
    shuffle=True, num_workers=4, drop_last=True, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg.val_batch_size,
    shuffle=False, num_workers=4, drop_last=True, pin_memory=True
)

print("Building model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepCrack()
model = torch.nn.DataParallel(model).to(device)
trainer = DeepCrackTrainer(model).to(device)

# AMP Scaler (universal)
try:
    scaler = GradScaler(device_type="cuda")
except TypeError:
    scaler = GradScaler(enabled=True)


def main():
    if cfg.pretrained_model:
        ckpt = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model.load_state_dict(ckpt, strict=False)

    try:
        for epoch in range(1, cfg.epoch + 1):
            # ---------------- Training ----------------
            model.train()
            train_losses = []

            pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Train')
            for img, lab in pbar:
                img = img.to(device, non_blocking=True)
                lab = lab.to(device, non_blocking=True)

                trainer.optimizer.zero_grad(set_to_none=True)

                with autocast("cuda" if NEW_AMP else torch.cuda.amp.autocast):
                    _ = trainer.train_op(img, lab)
                    loss = trainer.log_loss['total_loss']

                scaler.scale(loss).backward()
                scaler.step(trainer.optimizer)
                scaler.update()

                train_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())

                del img, lab, loss
                torch.cuda.empty_cache()

            print(f'Epoch {epoch} - Avg Train Loss: {np.mean(train_losses):.4f}')

            # ---------------- Validation ----------------
            model.eval()
            val_losses = []
            val_accs = {}

            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Val')
                for img, lab in pbar:
                    img = img.to(device)
                    lab = lab.to(device)

                    with autocast("cuda" if NEW_AMP else torch.cuda.amp.autocast):
                        preds = trainer.val_op(img, lab)
                        loss = trainer.log_loss['total_loss']

                    val_losses.append(loss.item())
                    acc = trainer.acc_op(preds[0], lab)

                    # Dynamically handle unknown metric keys
                    for k, v in acc.items():
                        if k not in val_accs:
                            val_accs[k] = 0.0
                        val_accs[k] += v

                    del img, lab
                    torch.cuda.empty_cache()

            n_val = len(val_loader)
            avg_val_loss = np.mean(val_losses)
            avg_acc = {k: v / n_val for k, v in val_accs.items()}

            print(f'Epoch {epoch} - Avg Val Loss: {avg_val_loss:.4f}')
            print(f'Epoch {epoch} - Validation Metrics:')
            for k, v in avg_acc.items():
                print(f'  {k}: {v:.5f}')

            # ---------------- Save Best ----------------
            main_acc_key = list(avg_acc.keys())[0] if avg_acc else None
            if main_acc_key and avg_acc[main_acc_key] > cfg.save_acc:
                cfg.save_acc = avg_acc[main_acc_key]
                tag = f'{cfg.name}_epoch({epoch})_acc({cfg.save_acc:.5f})'
                trainer.saver.save(model, tag=tag)
                print(f'✅ Best checkpoint saved: {tag}')

            trainer.saver.save(model, tag=f'{cfg.name}_epoch({epoch})')

    except KeyboardInterrupt:
        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n⚠️ Interrupted – model auto-saved.')
        sys.exit(0)


if _name_ == '_main_':
    main()