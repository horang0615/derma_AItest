import os
import torch
from torchvision import transforms


# 0) Project Root (절대경로 꼬임 방지)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 1) Track Switch
TRACK_NAME = "TrackD"   # "TrackA" "TrackB" "TrackC" "TrackD"
EXP_NAME = "external_bcc15"

TRACK_DATA_MAP = {
    "TrackA": "TrackA_binary_MEL_MN",
    "TrackB": "TrackB_binary_AK_MN",
    "TrackC": "TrackC_binary_BCC_MN",
    "TrackD": "TrackD_binary_AK_BCC",
}

# 2) Dataset Paths (기본: 내부 데이터)
TRACK_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", TRACK_DATA_MAP[TRACK_NAME])

TRAIN_DIR = os.path.join(TRACK_DATA_DIR, "train")
VAL_DIR   = os.path.join(TRACK_DATA_DIR, "val")
TEST_DIR  = os.path.join(TRACK_DATA_DIR, "test")   # 기본값

# 3) External Test Override (있으면 TEST_DIR 덮어씀)
EXTERNAL_TEST_DIR = "data/processed/external_labeled/TrackD_AK_BCC/test"
# EXTERNAL_TEST_DIR = "data/processed/external_labeled/TrackC_BCC_MN/test"
# EXTERNAL_TEST_DIR = ""  # 내부 test로 돌릴 땐 빈 문자열로

if EXTERNAL_TEST_DIR:
    TEST_DIR = os.path.join(BASE_DIR, EXTERNAL_TEST_DIR)

# 4) Run / Output Paths (트랙별 분리)
RUN_DIR = os.path.join(BASE_DIR, "runs", TRACK_NAME, EXP_NAME)
TRAIN_LOG_DIR = os.path.join(RUN_DIR, "train_logs")
TEST_LOG_DIR  = os.path.join(RUN_DIR, "test_logs")
MODEL_DIR     = os.path.join(BASE_DIR, "runs", TRACK_NAME, "model") 

# 5) Train Settings
NUM_CLASSES = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 6) Transform
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 7) Class Names (0/1 폴더 기준)

CLASS_NAMES_MAP = {
    "TrackA": ["MN", "MEL"],   # 0=MN, 1=MEL
    "TrackB": ["AK", "MN"],    # 0=AK, 1=MN
    "TrackC": ["BCC", "MN"],   # 0=BCC, 1=MN
    "TrackD": ["AK", "BCC"],   # 0=AK, 1=BCC
}
CLASS_NAMES = CLASS_NAMES_MAP[TRACK_NAME]

print("[CONFIG]", TRACK_NAME, "| TEST_DIR =", TEST_DIR)
