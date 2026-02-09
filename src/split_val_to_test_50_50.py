import os
import random
import shutil

random.seed(42)

BASE_DIR = "../data/processed/TrackA_binary_MEL_MN"
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

CLASSES = ["0_MN", "1_MEL"]
N_MOVE_PER_CLASS = 50  # 각 클래스에서 test로 이동

for cls in CLASSES:
    val_cls_dir = os.path.join(VAL_DIR, cls)
    test_cls_dir = os.path.join(TEST_DIR, cls)

    files = os.listdir(val_cls_dir)
    if len(files) < N_MOVE_PER_CLASS:
        raise ValueError(f"{cls} 데이터 부족: {len(files)}")

    selected = random.sample(files, N_MOVE_PER_CLASS)

    for fname in selected:
        src = os.path.join(val_cls_dir, fname)
        dst = os.path.join(test_cls_dir, fname)
        shutil.move(src, dst)

    print(f"[OK] {cls}: {len(selected)} files moved to test")
