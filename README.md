# Strawberry Detection — 추론 파이프라인

[딸기 병해 진단 팀 프로젝트](https://github.com/Kwakjaemin1007/Like-lion_-Smart-farm)에서 제가 맡은 추론 코드만 따로 정리한 레포입니다.
학습 노트북과 실험 결과는 팀 레포에 있습니다.

```
이미지 → [YOLO11s] 잎·과실 탐지 → 영역 잘라내기 → [EfficientNet-B0] 5종 병해 분류 → 결과 이미지 저장
```

## 실행

```bash
pip install -r requirements.txt
python inference.py   # 파일 상단의 YOLO_PATH, CLS_PATH, 데이터 경로를 환경에 맞게 수정
```

현재 경로는 Google Colab + Google Drive 기준입니다.

`PyTorch` `Ultralytics YOLO` `torchvision` `OpenCV`
