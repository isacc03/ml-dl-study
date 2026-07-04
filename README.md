# ML/DL Study

머신러닝과 딥러닝을 공부하며 작성한 노트북, 실습 코드, 정리 노트를 모아두는 저장소입니다.

## Structure

- `notebooks/`: Colab 또는 Jupyter 실습 노트북
- `scripts/`: 노트북에서 추출한 Python 코드
- `notes/`: 개념 정리, 실험 메모, 회고
- `data/`: 데이터 위치 안내만 보관하고 실제 데이터 파일은 Git에 올리지 않음

## Current Notes

| Topic | Notebook | Script | Source |
| --- | --- | --- | --- |
| 신경망 모델 훈련 | `notebooks/deep-learning/07-3_neural_network_training.ipynb` | `scripts/deep-learning/07-3_neural_network_training.py` | [Colab](https://colab.research.google.com/drive/1_NKY6AH8dSi-gtJ0VVImct5X7VdRhrMa?usp=sharing) |
| 합성곱 신경망을 사용한 이미지 분류 | `notebooks/deep-learning/08-2_convolutional_neural_network_image_classification.ipynb` | `scripts/deep-learning/08-2_convolutional_neural_network_image_classification.py` | [Colab](https://colab.research.google.com/drive/1qb1_cyo-QTk8QDj6NjMsj3qU8rz79tCV?usp=sharing) |

## Study Log Convention

새로운 학습 내용은 아래 흐름으로 추가합니다.

1. 노트북은 `notebooks/<topic>/NN_title.ipynb`에 저장합니다.
2. 재사용할 코드는 `scripts/<topic>/NN_title.py`로 분리합니다.
3. 핵심 개념, 실험 결과, 헷갈린 점은 `notes/<topic>.md`에 남깁니다.
4. 큰 데이터, 학습된 모델, 실행 결과 이미지는 Git에 직접 올리지 않습니다.
