import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from dataset import FolderBinaryDataset
from model import CustomEfficientNetB3Classifier
from configs.config import (
    DEVICE, TEST_DIR, TRANSFORM, BATCH_SIZE,
    MODEL_DIR, TEST_LOG_DIR, CLASS_NAMES, TRACK_NAME
)
from logger import Logger

from datetime import datetime
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
behavior_log_dir = os.path.join(TEST_LOG_DIR, f"behavior_{RUN_ID}")

print("[EVAL-BEHAVIOR CONFIG]", TRACK_NAME, "| TEST_DIR =", TEST_DIR, "| MODEL_DIR =", MODEL_DIR)


# 설정
LOW = 0.35
HIGH = 0.65


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_checkpoint(model, checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model


def _is_prob_output(t: torch.Tensor) -> bool:
    mn = float(t.min().item())
    mx = float(t.max().item())
    return (mn >= 0.0) and (mx <= 1.0)


def decide(prob: float) -> str:
    if prob < LOW:
        return "negative"
    elif prob > HIGH:
        return "positive"
    else:
        return "ambiguous"


def evaluate_behavior():
    # 준비
    behavior_log_dir = os.path.join(TEST_LOG_DIR, "behavior")
    _ensure_dir(behavior_log_dir)

    logger = Logger(behavior_log_dir, log_file="evaluation_behavior_log.json")

    logger.log({
        "message": "Behavior evaluation started",
        "LOW": LOW,
        "HIGH": HIGH,
        "TRACK_NAME": TRACK_NAME
    })

    test_dataset = FolderBinaryDataset(root_dir=TEST_DIR, transform=TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CustomEfficientNetB3Classifier().to(DEVICE)
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth.tar")
    model = _load_checkpoint(model, checkpoint_path, DEVICE)
    model.eval()

    # 평가
    results = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating (behavior)")):
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs) if not _is_prob_output(outputs) else outputs

            probs_np = probs.detach().cpu().numpy().astype(float)
            labels_np = labels.detach().cpu().numpy().astype(int)

            base = batch_idx * BATCH_SIZE
            for i in range(len(labels_np)):
                sample_idx = base + i
                if sample_idx >= len(test_dataset.ds.samples):
                    continue

                image_path = test_dataset.ds.samples[sample_idx][0]
                image_id = os.path.basename(image_path)

                prob = probs_np[i]
                decision = decide(prob)

                results.append({
                    "image_id": image_id,
                    "y_true": int(labels_np[i]),
                    "y_prob": float(prob),
                    "decision": decision,
                })

                all_probs.append(prob)
                all_labels.append(labels_np[i])

    df = pd.DataFrame(results)

    # 요약 지표
    amb_ratio = float((df["decision"] == "ambiguous").mean())
    coverage_ratio = float(1.0 - amb_ratio)

    if len(np.unique(all_labels)) > 1:
        auroc = float(roc_auc_score(all_labels, all_probs))
    else:
        auroc = None

    summary = {
        "LOW": LOW,
        "HIGH": HIGH,
        "num_samples": len(df),
        "ambiguous_ratio": amb_ratio,
        "coverage_ratio": coverage_ratio,
        "auroc_all": auroc,
    }

    # 저장
    df.to_csv(
        os.path.join(behavior_log_dir, "prediction_results_behavior.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([summary]).to_csv(
        os.path.join(behavior_log_dir, "metrics_summary_behavior.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    logger.log({"message": "Behavior evaluation completed", **summary})
    logger.save()

    print("Behavior evaluation done:", behavior_log_dir)


if __name__ == "__main__":
    evaluate_behavior()
