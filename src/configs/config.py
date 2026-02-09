import os
import torch
from torchvision import transforms

# ✅ config.py 파일 위치를 기준으로 프로젝트 루트 잡기 (경로 꼬임 방지)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))



# 1) 트랙 스위치

TRACK_NAME = "TrackA"   # "TrackB" "TrackC", "TrackD" 

# ✅ 데이터 폴더명 매핑 (네 실제 폴더명에 맞춰 적어줘)
TRACK_DATA_MAP = {   
    # "TrackB": "TrackB_binary_AK_MN",
    # "TrackC": "TrackC_binary_BCC_MN",
    # "TrackD": "TrackD_binary_AK_BCC",
    "TrackA": "TrackA_binary_MEL_MN",
}

TRACK_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", TRACK_DATA_MAP[TRACK_NAME])

TRAIN_DIR = os.path.join(TRACK_DATA_DIR, "train")
VAL_DIR   = os.path.join(TRACK_DATA_DIR, "val")
TEST_DIR  = os.path.join(TRACK_DATA_DIR, "test")

# 결과 저장도 트랙별 분리
RUN_DIR = os.path.join(BASE_DIR, "runs", TRACK_NAME)
TRAIN_LOG_DIR = os.path.join(RUN_DIR, "train_logs")
TEST_LOG_DIR  = os.path.join(RUN_DIR, "test_logs")
MODEL_DIR     = os.path.join(RUN_DIR, "model")


# 2) 학습 설정
NUM_CLASSES = 1          # binary
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3) 전처리
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 4) 클래스 이름도 트랙별로 같이 스위치
CLASS_NAMES_MAP = {
    # "TrackB": ["AK", "MN"],   
    # "TrackC": ["BCC", "MN"],
    "TrackD": ["MN", "BCC"],
    "TrackA": ["MEL", "MN"],
}
CLASS_NAMES = CLASS_NAMES_MAP.get(TRACK_NAME, ["0", "1"])
