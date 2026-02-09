import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score
)

from dataset import FolderBinaryDataset
from model import CustomEfficientNetB3Classifier
from configs.config import DEVICE, TEST_DIR, TRANSFORM, BATCH_SIZE, MODEL_DIR, TEST_LOG_DIR, CLASS_NAMES, TRACK_NAME

from logger import Logger
print("[EVAL CONFIG]", TRACK_NAME, "| TEST_DIR =", TEST_DIR, "| MODEL_DIR =", MODEL_DIR)



# ====== 설정 ======
THRESHOLD = 0.5

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_checkpoint(model, checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # 혹시 state_dict 자체로 저장된 경우 대비
        model.load_state_dict(ckpt)
    return model


def _is_prob_output(t: torch.Tensor) -> bool:
    # 출력이 확률(0~1) 범위인지 대략 판별
    mn = float(t.min().item())
    mx = float(t.max().item())
    return (mn >= 0.0) and (mx <= 1.0)


def evaluate_binary_model():
    # ====== 준비 ======
    _ensure_dir(TEST_LOG_DIR)
    logger = Logger(TEST_LOG_DIR, log_file="evaluation_log.json")

    logger.log({"message": "Loading test dataset...", "TEST_DIR": TEST_DIR})
    test_dataset = FolderBinaryDataset(root_dir=TEST_DIR, transform=TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.log({"message": "Test dataset loaded.", "num_samples": len(test_dataset), "batch_size": BATCH_SIZE})

    logger.log({"message": "Loading model..."})
    model = CustomEfficientNetB3Classifier().to(DEVICE)
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth.tar")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = _load_checkpoint(model, checkpoint_path, DEVICE)
    model.eval()
    logger.log({"message": "Model loaded.", "checkpoint_path": checkpoint_path, "device": DEVICE})

    # ====== 평가 ======
    all_labels: list[int] = []
    all_probs: list[float] = []
    all_image_ids: list[str] = []
    all_results = []

    logger.log({"message": "Starting evaluation..."})

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # dataset이 (img, label)만 반환하는 구조
            images, labels = batch[0], batch[1]
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images).squeeze()

            # logits이면 sigmoid 적용
            if not _is_prob_output(outputs):
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs

            preds = (probs >= THRESHOLD).long()

            labels_np = labels.detach().cpu().numpy().astype(int)
            probs_np = probs.detach().cpu().numpy().astype(float)
            preds_np = preds.detach().cpu().numpy().astype(int)

            all_labels.extend(labels_np.tolist())
            all_probs.extend(probs_np.tolist())

            # image_id 추출: ImageFolder는 samples에 (path, label) 들고 있음
            # 현재 배치가 dataset에서 몇 번째 인덱스인지 계산
            base = batch_idx * BATCH_SIZE
            for i in range(len(labels_np)):
                sample_idx = base + i
                # 마지막 배치에서 sample_idx가 넘어갈 수 있으니 방어
                if sample_idx >= len(test_dataset.ds.samples):
                    continue
                image_path = test_dataset.ds.samples[sample_idx][0]
                image_id = os.path.basename(image_path)

                all_image_ids.append(image_id)

                all_results.append({
                    "image_id": image_id,
                    "y_true": int(labels_np[i]),
                    "y_pred": int(preds_np[i]),
                    "y_prob": float(probs_np[i]),
                    "actual": CLASS_NAMES[int(labels_np[i])],
                    "predicted": CLASS_NAMES[int(preds_np[i])],
                    "correct": int(labels_np[i] == preds_np[i]),
                })

    y_true = np.array(all_labels, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ====== 지표 ======
    if len(np.unique(y_true)) < 2:
        # test에 한 클래스만 있으면 auc 계산 불가
        auc_score = None
        logger.log({"message": "Warning: only one class present in y_true; AUROC cannot be computed."})
    else:
        auc_score = float(roc_auc_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)

    logger.log({
        "message": "Metrics calculated.",
        "threshold": THRESHOLD,
        "auroc": auc_score,
        "num_samples": int(len(y_true)),
        "pos_rate_true": float(np.mean(y_true == 1)),
        "pos_rate_pred": float(np.mean(y_pred == 1)),
    })

    # ====== 저장: PNG ======
    cm_path = os.path.join(TEST_LOG_DIR, "confusion_matrix.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    roc_path = os.path.join(TEST_LOG_DIR, "roc_curve.png")
    roc_values_path = os.path.join(TEST_LOG_DIR, "roc_values.csv")

    if auc_score is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(roc_path, dpi=200)
        plt.close()

        # ROC values + precision by threshold
        precision_scores = [precision_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
        pd.DataFrame({
            "Threshold": thresholds,
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
            "Precision": precision_scores
        }).to_csv(roc_values_path, index=False, encoding="utf-8-sig")
    else:
        # 빈 파일 대신 안내용 파일
        pd.DataFrame([{"message": "AUROC not computed because y_true has only one class in test set."}]).to_csv(
            roc_values_path, index=False, encoding="utf-8-sig"
        )
        # roc_curve.png는 생성하지 않음

    # ====== 저장: CSV ======
    # (1) 사람용 결과
    prediction_results_path = os.path.join(TEST_LOG_DIR, "prediction_results.csv")
    pd.DataFrame(all_results)[["image_id", "actual", "predicted", "y_prob", "correct"]].to_csv(
        prediction_results_path, index=False, encoding="utf-8-sig"
    )

    # (2) 태블로용 정형 결과
    prediction_tableau_path = os.path.join(TEST_LOG_DIR, "prediction_tableau.csv")
    pd.DataFrame(all_results)[["image_id", "y_true", "y_pred", "y_prob", "correct"]].to_csv(
        prediction_tableau_path, index=False, encoding="utf-8-sig"
    )

    # (3) 메트릭 요약
    metrics_summary_path = os.path.join(TEST_LOG_DIR, "metrics_summary.csv")
    pd.DataFrame([{
        "model_name": "main_model",
        "split": "test",
        "threshold": THRESHOLD,
        "auroc": auc_score,
        "num_samples": int(len(y_true)),
    }]).to_csv(metrics_summary_path, index=False, encoding="utf-8-sig")

    logger.log({"message": "Saved outputs", "TEST_LOG_DIR": TEST_LOG_DIR})
    logger.log({"message": "Paths", "paths": {
        "confusion_matrix_png": cm_path,
        "roc_curve_png": roc_path if auc_score is not None else None,
        "roc_values_csv": roc_values_path,
        "prediction_results_csv": prediction_results_path,
        "prediction_tableau_csv": prediction_tableau_path,
        "metrics_summary_csv": metrics_summary_path,
    }})
    logger.log({"message": "Evaluation completed."})
    logger.save()

    print("✅ Done! Outputs saved to:", TEST_LOG_DIR)


if __name__ == "__main__":
    evaluate_binary_model()
