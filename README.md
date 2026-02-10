피부질환 이미지 이진 분류 실험 (AI Hub 합성데이터 기반)

(English TL;DR) Binary classification experiments using AI Hub synthetic skin lesion images.
Trained/evaluated on original split (train/val/test) and ran extra evaluation on external images for Track C & D.

1. 프로젝트 소개
이 프로젝트는 AI Hub 피부질환 합성 이미지 데이터를 이용해 피부 병변을 이진 분류(Binary Classification)
하는 모델을 학습하고 평가한 기록.

목표: 
  - Track A~D로 정의된 2개 클래스 조합에 대해 이진 분류 모델 학습 및 평가
  - 동일한 파이프라인을 유지한 상태에서 데이터 구성 변화(외부 이미지 추가)가 모델 성능에 미치는 영향 관찰
모델: EfficientNet-B3 기반 이진 분류 (
출력: sigmoid 기반 확률값(0~1)
결과물: ROC Curve, Confusion Matrix, 이미지별 예측 결과 CSV, Tableau 분석용 정형 CSV

Track 정의: 
Track A: 0_MN, 1_MEL / Track B: 0_AK, 1_MN / Track C: 0_BCC, 1_MN / Track D: 0_AK, 1_BCC

2. 실험 흐름 (커밋 기준 요약)
(1) 1차: Original Test (첫 번째 커밋_2026.02.09_model_fin_01)
- 목적
기본 학습/평가 파이프라인이 정상 동작하는지 확인, Track별 baseline 성능 확보

- 방법
AI Hub 데이터만 사용해 고정 split으로 실험(Train: 800장, Val: 50장, Test: 50장)
Track A~D 모두 동일한 학습/평가 파이프라인을 사용, config 파일에서 Track만 변경하여 반복 실행

(2) 2차: External 이미지 파생 실험 (두 번째 커밋_2026.02.09_extra_img_20)
- 목적
  합성데이터가 아닌 실전 환경의 외부 이미지를 적용했을 때 모델 성능의 변화 확인

- 방법
original test 이후, 별도로 수집한 외부 BCC 이미지 20장(라벨링 완료)를 수집
외부 이미지는 기존 test dataset과 별개로 평가용 test dataset으로 분리
해당 외부 이미지를 활용해 Track C(BCC vs MN), Track D(AK vs BCC)에 대해 추가 평가 수행
