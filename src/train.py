import os
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dataset import FolderBinaryDataset
from configs.config import (
    TRAIN_DIR, VAL_DIR,
    TRANSFORM, BATCH_SIZE,
    DEVICE, MODEL_DIR, TRAIN_LOG_DIR,
    LEARNING_RATE, EPOCHS, SEED
)
from logger import Logger
from model import CustomEfficientNetB3Classifier



# 0) 재현성 고정

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 완전 고정(조금 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1) 저장 유틸
def save_checkpoint(state, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")


# 2) AUROC (예외 안전)
def safe_auroc(prob_list, label_list):
    """
    prob_list: list[float] (0~1)
    label_list: list[int] (0/1)
    """
    # 한 클래스만 있으면 roc_auc_score가 에러남
    if len(set(label_list)) < 2:
        return float("nan")
    return roc_auc_score(np.array(label_list), np.array(prob_list))


def save_auroc_data(outputs, labels, epoch, phase, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data = {
        "epoch": epoch,
        "phase": phase,
        "outputs": [round(float(o), 6) for o in outputs],
        "labels": [int(x) for x in labels],
    }
    file_path = os.path.join(save_dir, f"{phase}_auroc_data_epoch_{epoch}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"AUROC data saved for {phase} epoch {epoch} -> {file_path}")


# 3) 학습/검증
def train_and_validate(max_epochs=EPOCHS, patience=20, save_every=5):
    set_seed(SEED)

    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger = Logger(TRAIN_LOG_DIR, "train_log.json")
    logger.log({
        "message": "Training started",
        "TRAIN_DIR": TRAIN_DIR,
        "VAL_DIR": VAL_DIR,
        "DEVICE": DEVICE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": max_epochs,
        "SEED": SEED,
    })

    train_dataset = FolderBinaryDataset(TRAIN_DIR, transform=TRANSFORM)
    val_dataset   = FolderBinaryDataset(VAL_DIR, transform=TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("train samples:", len(train_dataset), "val samples:", len(val_dataset))
    print("class_to_idx:", train_dataset.ds.class_to_idx)

    # binary는 항상 1 출력(로짓 1개)
    model = CustomEfficientNetB3Classifier(num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    total_start_time = time.time()
    best_val_loss = float("inf")
    best_epoch = 0
    early_stopping_counter = 0

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # ---------- Train ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_probs = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Train {epoch+1}/{max_epochs}"):
            images, labels = batch[0], batch[1]

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images).squeeze(1)              # [B]
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(logits)                 # [B]
            preds = (probs >= 0.5).long()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_probs.extend(probs.detach().cpu().tolist())
            train_labels.extend(labels.detach().cpu().tolist())

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        train_auc = safe_auroc(train_probs, train_labels)

        save_auroc_data(train_probs, train_labels, epoch + 1, "train", TRAIN_LOG_DIR)

        # ---------- Val ----------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                images, labels = batch[0], batch[1]
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images).squeeze(1)
                loss = criterion(logits, labels.float())
                val_loss_sum += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_probs.extend(probs.detach().cpu().tolist())
                val_labels.extend(labels.detach().cpu().tolist())

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        val_auc = safe_auroc(val_probs, val_labels)

        save_auroc_data(val_probs, val_labels, epoch + 1, "val", TRAIN_LOG_DIR)

        # ---------- Logging ----------
        print(
            f"[Epoch {epoch+1}/{max_epochs}] "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} AUROC {train_auc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUROC {val_auc:.4f} | "
            f"Time {time.time() - epoch_start:.1f}s"
        )

        logger.log({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_auroc": None if np.isnan(train_auc) else float(train_auc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_auroc": None if np.isnan(val_auc) else float(val_auc),
        })

        # ---------- Best checkpoint ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0

            save_checkpoint({
                "epoch": best_epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(best_val_loss),
            }, MODEL_DIR, "best_model.pth.tar")
        else:
            early_stopping_counter += 1

        # periodic save
        if (epoch + 1) % save_every == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_loss),
            }, MODEL_DIR, f"checkpoint_epoch_{epoch + 1}.pth.tar")

        scheduler.step(val_loss)

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.1f}s")
    logger.log({"message": "Training completed", "best_epoch": best_epoch, "best_val_loss": float(best_val_loss)})
    logger.save()


if __name__ == "__main__":
    print("Starting training...")
    train_and_validate()
    print("Training complete.")
