# wafer-map-defect-classification
WM-811K wafer defect classification with lot-aware split and class imbalance analysis

## Overview

WM-811K wafer map 데이터를 이용하여
9개 defect pattern을 분류하고
class imbalance 처리 방법을 비교하였다.

## Dataset

- Total wafers: 811,457
- Labeled wafers: 172,950
- 9 classes
- Normal class: 85.25%

## Data Split

Lot-aware split

Train: 120,979
Validation: 26,116
Test: 25,855

Lot overlap = 0
 
## Experiments

| Model | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Baseline CNN | 0.9663 | 0.7851 | 0.7745 |
| Full Weighted CNN | 0.9330 | 0.8356 | 0.7534 |
| Tempered CNN | 0.9670 | 0.7748 | 0.7861 |
| Tempered + Augmentation | 0.9660 | 0.8133 | 0.7655 |

## Key Findings

- Baseline CNN achieved high overall accuracy, but Scratch recall was only 3.95%.
- Full class weighting improved minority-class recall, but reduced overall Accuracy and Macro-F1.
- Tempered weighting using sqrt(class weight) achieved the highest Macro-F1.
- Rotation/flip augmentation improved Scratch recall and Balanced Accuracy, but did not improve Macro-F1.

## Current Best Model

**Tempered CNN**

- Accuracy: 96.70%
- Balanced Accuracy: 77.48%
- Macro-F1: 78.61%

## Next Steps

- Grad-CAM visualization
- Error analysis
- Defect pattern interpretation
- Streamlit demo

# WM-811K Wafer Defect Classification

## 1. Project Overview

본 프로젝트는 WM-811K wafer map 데이터를 이용하여
반도체 wafer의 불량 패턴을 분류하고,
심한 클래스 불균형(class imbalance)이 모델 성능에 미치는 영향을 분석하는 것을 목표로 한다.

단순히 높은 Accuracy를 얻는 것을 목표로 하지 않고,
실제 제조 데이터 분석에서 중요한 다음 문제들을 함께 다루었다.

- 동일 생산 Lot에 의한 데이터 누수(data leakage) 방지
- 클래스 불균형에 강건한 평가 지표 사용
- Class Weight에 따른 minority class 성능 변화 분석
- 과도한 weighting을 완화한 tempered weighting 실험
- Rotation / Flip 기반 augmentation 실험
- Grad-CAM을 이용한 공간적 판단 영역 분석
- Scratch class의 오분류 패턴 분석

최종적으로 모델 성능뿐 아니라,
모델이 어떤 패턴에서 강하고 어떤 패턴에서 실패하는지까지 분석하는 것을 목표로 하였다.


---

# 2. Dataset

사용 데이터셋:

**WM-811K Wafer Map Dataset**

원본 데이터는 `LSWMD.pkl` 형태로 제공되며,
wafer map과 생산 lot, wafer index, failure type 등의 정보를 포함한다.

확인된 주요 컬럼은 다음과 같다.

- `waferMap`
- `dieSize`
- `lotName`
- `waferIndex`
- `trianTestLabel`
- `failureType`

주의할 점은 원본 데이터에
`trainTestLabel`이 아니라 `trianTestLabel`이라는 컬럼명으로 저장되어 있다는 점이다.


## Dataset Size

전체 wafer:

```text
811,457

failureType이 실제로 존재하는 labeled wafer:
172,950

label이 없는 wafer:
638,507

생산 lot:
46,293

Label Structure
failureType은 일반 문자열이 아니라
numpy.ndarray 형태로 저장되어 있었다.
예:
array([['none']])


따라서 모델 학습 전 별도의 label 정제 과정이 필요했다.
Important Label Distinction
데이터에서 다음 두 값은 반드시 구분해야 한다.
None

은 failureType 자체가 존재하지 않는 unlabeled wafer를 의미한다.
반면
"none"

은 정상 wafer class를 의미한다.
따라서 unlabeled wafer와 normal wafer를 동일하게 처리하면 안 된다.
3. Classes
본 프로젝트에서는 총 9개의 클래스를 사용하였다.
0 = none
1 = Center
2 = Donut
3 = Edge-Loc
4 = Edge-Ring
5 = Loc
6 = Random
7 = Scratch
8 = Near-full

라벨된 데이터의 class distribution은 다음과 같다.
Class	Count
none	147,431
Edge-Ring	9,680
Edge-Loc	5,189
Center	4,294
Loc	3,593
Scratch	1,193
Random	866
Donut	555
Near-full	149


정상 클래스인 none이 전체 labeled data의 약 **85.25%**를 차지한다.
따라서 단순 Accuracy만으로는 모델 성능을 제대로 평가하기 어렵다.
예를 들어 모든 wafer를 none으로 예측하더라도
약 85%에 가까운 Accuracy가 나올 수 있다.
이 때문에 본 프로젝트에서는 다음 지표를 함께 사용하였다.
- Accuracy
- Balanced Accuracy
- Macro-F1
- Class-wise Recall
- Confusion Matrix
4. Project Progress by Day
Day 1 — Dataset Loading and Structure Inspection
Goal
첫날의 목표는 모델 학습이 아니라
데이터셋 자체를 정확히 이해하는 것이었다.
진행한 작업:
- WM-811K 다운로드
- LSWMD.pkl 로딩
- DataFrame structure 확인
- waferMap shape 확인
- failureType structure 확인
- lotName 확인
- unlabeled wafer / normal wafer 구분
Key Findings
원본 DataFrame shape:
811,457 rows
6 columns

첫 wafer map의 예:
shape = (45, 48)

waferMap은 각 die 상태를 나타내는 2D array 형태였다.
주요 값은 다음과 같이 해석할 수 있다.
0 = wafer background
1 = normal die
2 = defect die

wafer map은 일반 사진이 아니라
범주형 2D grid 데이터라는 점이 중요하다.
Lot Structure
확인된 unique lot 수:
46,293

lot당 wafer 수 통계:
mean   ≈ 17.63
median = 24
max    = 25

동일 lot에 여러 wafer가 존재하기 때문에
단순 random split을 사용할 경우
동일 생산 lot의 wafer가 Train과 Test에 동시에 포함될 가능성이 있다.
이 문제는 이후 lot-aware split의 근거가 되었다.
Day 1 Conclusion
Total wafers    : 811,457
Labeled wafers  : 172,950
Unlabeled       : 638,507
Unique lots     : 46,293

가장 중요한 데이터 처리 원칙은
None != "none"

이라는 점이다.
Day 2 — EDA and Preprocessing
Goal
wafer map의 공간적 구조와 class distribution을 확인하고,
CNN 입력을 위한 전처리 방식을 결정하였다.
진행한 작업:
- 9개 class 예시 시각화
- wafer shape 분포 확인
- square / non-square wafer 확인
- resize / padding 비교
- label encoding
- class imbalance 분석
Wafer Shape Problem
wafer map의 크기는 모두 동일하지 않았다.
예:
45 × 48
26 × 26
32 × 29
...

CNN은 batch 단위 입력에서 동일한 spatial size가 필요하므로
입력 크기를 통일해야 했다.
Resize vs Padding
두 가지 방식을 비교하였다.
Method A
64 × 64 nearest resize

Method B
aspect-ratio preserving resize
+
padding

waferMap은 일반 이미지가 아니라
0 / 1 / 2로 구성된 범주형 grid이기 때문에
bilinear interpolation을 사용하지 않았다.
대신
cv2.INTER_NEAREST


를 사용하여
의미 없는 중간 pixel value가 생성되지 않도록 하였다.
Final Baseline Input
baseline에서는 다음 입력 형식을 사용하였다.
1 × 64 × 64

pixel normalization:
0 → 0.0
1 → 0.5
2 → 1.0

Class Imbalance Observation
정상 none class가 labeled data의 약 85%를 차지함을 확인했다.
이 결과는 이후 실험 설계의 핵심이 되었다.
Day 2 Conclusion
- Input size: 64 × 64
- Interpolation: nearest-neighbor
- Number of classes: 9
- Strong class imbalance confirmed
- Accuracy alone is not sufficient
Day 3 — Lot-Aware Train / Validation / Test Split
Goal
같은 생산 lot이 Train / Validation / Test에 동시에 포함되지 않도록
lot-aware split을 구성하였다.
단순 wafer-level random split 대신
lotName을 group으로 사용하였다.
Split Strategy
Train      ≈ 70%
Validation ≈ 15%
Test       ≈ 15%

실제 결과:
Split	Samples	Ratio
Train	120,979	69.95%
Validation	26,116	15.10%
Test	25,855	14.95%


Leakage Check
다음 조건을 직접 검증하였다.
Train ∩ Validation = 0
Train ∩ Test       = 0
Validation ∩ Test  = 0

실제 결과:
Train ∩ Val  : 0
Train ∩ Test : 0
Val ∩ Test   : 0

따라서 동일 생산 lot이 서로 다른 split에 동시에 존재하지 않는다.
Class Distribution after Split
lot 기준으로 데이터를 분할했음에도
Train / Validation / Test의 class distribution은 크게 무너지지 않았다.
예:
none
Train ≈ 85.22%
Val   ≈ 85.25%
Test  ≈ 85.35%

Rare Class Limitation
Test set에서 Near-full class는 단 15개만 존재하였다.
Near-full Test support = 15

따라서 Near-full class의 recall은
몇 개의 sample만 바뀌어도 크게 변할 수 있다.
이 class의 성능은 강하게 일반화해서 해석하지 않도록 주의하였다.
PyTorch Data Pipeline
최종 CNN input batch shape:
[128, 1, 64, 64]

구성:
DataFrame
→ WaferDataset
→ resize
→ normalization
→ Tensor
→ DataLoader

Day 3 Conclusion
본 프로젝트에서는
wafer-level random split이 아닌
lot-aware split을 사용하였다.
이는 동일 lot에서 발생할 수 있는 유사 공정 조건이
Train과 Test에 동시에 유입되는 것을 방지하기 위함이다.
Day 4 — Baseline CNN
Goal
클래스 불균형 처리를 적용하지 않은
Simple CNN baseline을 구축하였다.
이 baseline은 이후 weighting 실험의 기준점으로 사용하였다.
Model Architecture
구성:
Conv2D
BatchNorm
ReLU
MaxPool

Conv2D
BatchNorm
ReLU
MaxPool

Conv2D
BatchNorm
ReLU
MaxPool

Flatten

Linear
ReLU
Dropout

Linear

입력:
1 × 64 × 64

출력:
9 classes

Training Setup
Loss       : CrossEntropyLoss
Optimizer  : Adam
Learning Rate : 1e-3
Batch Size : 128
Epochs     : 10

Class weight는 적용하지 않았다.
Best Model Selection
데이터 불균형이 매우 심하기 때문에
best model은 Accuracy가 아닌
Validation Macro-F1

기준으로 선택하였다.
Best Validation Result
best epoch:
Epoch 7

Validation:
Accuracy          ≈ 0.9642
Macro-F1          ≈ 0.7620
Balanced Accuracy ≈ 0.7649

Test Result
Accuracy          : 0.9663
Balanced Accuracy : 0.7851
Macro-F1          : 0.7745
Weighted F1       : 0.9632

Class-wise Recall
Class	Recall
none	0.9881
Center	0.9602
Donut	0.8675
Edge-Loc	0.8290
Edge-Ring	0.9832
Loc	0.5851
Random	0.9466
Scratch	0.0395
Near-full	0.8667


Major Finding
전체 Accuracy는 96.63%로 매우 높았지만,
Scratch recall은 단 **3.95%**였다.
즉 높은 Accuracy가
모든 defect class를 잘 분류한다는 의미는 아니었다.
특히 class imbalance가 심한 문제에서는
Accuracy만 보면 모델 성능을 과대평가할 수 있음을 확인하였다.
Day 4 Conclusion
Baseline CNN은
Accuracy = 96.63%
Macro-F1 = 0.7745

를 기록했지만,
Scratch Recall = 3.95%
Loc Recall     = 58.51%

로 일부 minority class의 성능이 매우 낮았다.
이 결과를 바탕으로
다음 실험에서 class weighting을 적용하였다.
Day 5 — Full Class Weight
Goal
minority defect class의 낮은 recall을 개선하기 위해
class-weighted CrossEntropyLoss를 적용하였다.
Class Weights
Train set 기준 balanced class weight:
Class	Weight
none	0.1304
Center	4.4688
Donut	36.7271
Edge-Loc	3.7031
Edge-Ring	1.9564
Loc	5.5753
Random	21.8927
Scratch	15.4330
Near-full	125.6272


Near-full의 weight가 약 125.6으로
매우 큰 값을 가지는 것을 확인했다.
Training Observation
weighted model에서는
초기 Accuracy가 baseline보다 크게 낮았다.
이는 majority class인 none의 영향력이 감소하고
minority class에 훨씬 큰 penalty를 부여했기 때문이다.
또한 Validation Accuracy와 Loss가
epoch별로 크게 변동하는 현상도 확인하였다.
이는 극단적인 class weight가
학습 안정성에 영향을 줄 가능성을 시사한다.
Test Result
Accuracy          : 0.9330
Balanced Accuracy : 0.8356
Macro-F1          : 0.7534

Recall Changes
Baseline → Full Weighted
none
0.9881 → 0.9463

Donut
0.8675 → 0.9759

Edge-Loc
0.8290 → 0.8536

Loc
0.5851 → 0.5920

Random
0.9466 → 0.9542

Scratch
0.0395 → 0.3446

Near-full
0.8667 → 0.9333

Major Finding
Scratch recall:
3.95%
→
34.46%

로 크게 개선되었다.
반면:
Accuracy
96.63%
→
93.30%

Macro-F1:
0.7745
→
0.7534

로 감소하였다.
Interpretation
Full class weighting은
minority class recall을 크게 개선하였다.
그러나 다수 class 성능과 전체 classification quality를 일부 희생하였다.
특히 Balanced Accuracy는 증가했지만
Macro-F1은 오히려 감소하였다.
이는 minority recall을 높이는 과정에서
false positive가 증가했을 가능성을 의미한다.
Day 5 Conclusion
Full balanced weighting은
minority class detection에는 효과적이었지만
전체 성능 trade-off가 너무 컸다.
따라서 다음 실험에서는
극단적인 weight를 완화하기 위해
sqrt-based tempered weighting을 적용하였다.
Day 6 — Tempered Class Weight
Goal
Full balanced weighting의 과도한 class weight를 완화하면서
minority class 성능 개선 효과를 유지할 수 있는지 확인하였다.
Method
기존 weight에 square root를 적용하였다.
tempered_weight = sqrt(class_weight)

예:
Near-full

125.63
→
sqrt
→
약 11.21

즉 rare class를 더 중요하게 보되
극단적으로 큰 gradient가 발생하는 문제를 완화하였다.
Test Result
Accuracy          : 0.9670
Balanced Accuracy : 0.7748
Macro-F1          : 0.7861

Comparison
Model	Accuracy	Balanced Accuracy	Macro-F1
Baseline	0.9663	0.7851	0.7745
Full Weighted	0.9330	0.8356	0.7534
Tempered	0.9670	0.7748	0.7861


Recall Changes
Baseline → Tempered
Scratch
0.0395 → 0.1977

Loc
0.5851 → 0.6649

Donut
0.8675 → 0.8795

일부 클래스는 감소하였다.
Center
0.9602 → 0.9104

Edge-Loc
0.8290 → 0.7850

Random
0.9466 → 0.9084

Major Finding
Tempered model은
Accuracy = 0.9670
Macro-F1 = 0.7861

로 현재 실험 중 가장 높은 Macro-F1을 기록하였다.
또한 Scratch / Loc 등 어려운 minority class 성능도
baseline보다 개선되었다.
Day 6 Conclusion
Full balanced weighting은
minority recall을 많이 개선했지만
전체 performance 손실이 컸다.
반면 sqrt weighting은
그 trade-off를 완화하였다.
따라서 본 프로젝트의 대표 모델로
Tempered CNN을 선택하였다.
Day 7 — Tempered Weight + Augmentation
Goal
Tempered weighting을 유지하면서
geometric augmentation을 추가했을 때
minority class의 generalization 성능이 더 개선되는지 확인하였다.
Augmentation
Train set에만 다음 augmentation을 적용하였다.
Random 90° rotation
Horizontal flip
Vertical flip

Validation / Test에는 augmentation을 적용하지 않았다.
Why Limited Augmentation?
wafer map은 일반 자연 이미지가 아니다.
pixel 값 자체가 die 상태를 의미하기 때문에
다음 augmentation은 사용하지 않았다.
- color jitter
- brightness change
- blur
- arbitrary interpolation
또한 90° rotation과 flip은
pixel 값 0 / 1 / 2를 그대로 유지할 수 있다는 장점이 있다.
Test Result
Accuracy          : 0.9660
Balanced Accuracy : 0.8133
Macro-F1          : 0.7655

Model Comparison
Model	Accuracy	Balanced Accuracy	Macro-F1
Baseline	0.9663	0.7851	0.7745
Full Weighted	0.9330	0.8356	0.7534
Tempered	0.9670	0.7748	0.7861
Tempered + Aug	0.9660	0.8133	0.7655


Recall Comparison
Class	Baseline	Full Weighted	Tempered	Tempered + Aug
none	0.9881	0.9463	0.9896	0.9885
Center	0.9602	0.9536	0.9104	0.9320
Donut	0.8675	0.9759	0.8795	0.9036
Edge-Loc	0.8290	0.8536	0.7850	0.7500
Edge-Ring	0.9832	0.9665	0.9707	0.9686
Loc	0.5851	0.5920	0.6649	0.6476
Random	0.9466	0.9542	0.9084	0.8626
Scratch	0.0395	0.3446	0.1977	0.3333
Near-full	0.8667	0.9333	0.6667	0.9333


Major Finding
Scratch recall:
Tempered
19.77%

→

Tempered + Aug
33.33%

로 크게 개선되었다.
Balanced Accuracy도:
0.7748
→
0.8133

으로 증가하였다.
하지만 Macro-F1은:
0.7861
→
0.7655

로 감소하였다.
Interpretation
Rotation / Flip augmentation은
일부 minority class의 recall에는 도움이 되었다.
그러나 모든 defect class에 동일하게 유효하지는 않았다.
특히 Edge-Loc, Random 등의 recall은 감소하였다.
가능한 해석은
일부 wafer defect pattern에서
절대적인 방향 또는 위치 정보 자체가
분류에 유용할 수 있다는 것이다.
단, 이는 실험 결과에 대한 추론이며
실제 공정 원인으로 단정하지 않는다.
Day 7 Conclusion
Augmentation은
Scratch와 일부 minority class의 recall을 개선했지만
전체 Macro-F1을 개선하지는 못했다.
따라서 최종 대표 모델은 계속
Tempered CNN으로 유지하였다.
Day 8 — Grad-CAM and Error Analysis
Goal
지금까지는 모델이
“얼마나 잘 맞히는가”를 평가하였다.
Day 8에서는
“모델이 wafer의 어느 공간 영역을 보고 판단하는가”를 분석하였다.
대표 모델:
Tempered CNN

Grad-CAM
Grad-CAM은 CNN의 convolution feature map과 gradient를 이용하여
특정 class prediction에 기여한 spatial region을 시각화하는 방법이다.
중요한 점은 Grad-CAM이
실제 공정 원인을 설명하는 방법은 아니라는 것이다.
Grad-CAM은 오직
모델이 분류 시 어느 공간 영역에 반응했는가

를 시각화한다.
Center Example
정분류 Center sample에서
Grad-CAM의 strongest activation이
실제 중앙 defect cluster와 거의 겹치는 것을 확인하였다.
True       : Center
Prediction : Center

관찰:
Central defect cluster
≈
Grad-CAM hotspot

이는 Center class 예측 시
CNN이 spatially meaningful한 feature를 사용하고 있음을 시사한다.
Scratch Performance Analysis
Tempered CNN에서 Test Scratch:
Total   : 177
Correct : 35
Wrong   : 142

Scratch recall:
35 / 177
≈
19.77%

Scratch Misclassification Distribution
Scratch 오분류 142개의 prediction:
Predicted Class	Count
none	80
Loc	58
Edge-Loc	4


즉 Scratch 오분류의 대부분은
none
+
Loc

에 집중되어 있었다.
Interpretation
Scratch는 선형 defect structure가 핵심 특징이다.
하지만 일부 wafer에서는
선형 defect가 약하거나 분산되어 보일 수 있으며,
이 경우 모델이 다음과 같이 판단했을 가능성이 있다.
weak / sparse pattern
→ none

local cluster-like pattern
→ Loc

이는 error pattern을 기반으로 한 추론이며,
실제 공정적 원인을 의미하지 않는다.
Scratch → none Grad-CAM
일부 Scratch → none 오분류 사례에서는
Grad-CAM activation이 특정 linear scratch region에 집중되지 않고
wafer 여러 영역으로 분산되는 현상이 관찰되었다.
이는 모델이 Scratch의 전체 선형 구조보다
산발적인 local feature를 사용했을 가능성을 시사한다.
Scratch → Loc Grad-CAM
Scratch → Loc 오분류 사례에서는
실제 long linear defect보다
wafer boundary 또는 일부 국소 영역에
activation이 집중되는 사례를 확인하였다.
이는 모델이 scratch의 전체 구조보다
local defect region을 강조하면서
Loc으로 오분류했을 가능성을 시사한다.
Grad-CAM Limitation
일부 correctly classified Scratch sample에서는
Grad-CAM이 거의 완전히 zero map으로 나타났다.
확인 결과:
CAM min  = 0
CAM max  = 0
CAM mean = 0

따라서 다음과 같은 점을 확인하였다.
Correct prediction
≠
Always meaningful Grad-CAM

즉 Grad-CAM 해석 가능성 자체도
모든 class와 sample에서 동일하게 신뢰할 수 있는 것은 아니다.
Additional Grad-CAM Debugging
standard Grad-CAM의 target layer를 바꾸어
마지막 convolution / ReLU layer를 비교하였다.
또한 raw CAM을 직접 확인하여
CAM collapse 원인을 분석하였다.
한 검사에서는:
Raw CAM min      : -0.0737
Raw CAM max      : 0.1067
Raw CAM mean     : 0.0089
Positive ratio   : 0.8320

가 확인되었다.
따라서 모든 zero-CAM 현상을 단순히
“모든 activation이 음수여서 ReLU에서 제거되었다”
라고 설명할 수 없음을 확인하였다.
또한 notebook 환경에서
image, row, wafer 같은 공통 변수명을 반복 사용하면서
분석 sample이 덮어써질 수 있다는 점도 확인하였다.
이후에는
scratch_image
scratch_row
scratch_wafer

처럼 명확한 변수명을 사용하여
sample mismatch 가능성을 줄였다.
Day 8 Conclusion
Grad-CAM 분석을 통해 다음을 확인하였다.
Successful Case
Center
→ 실제 중앙 defect region과 attention 일치

Error Cases
Scratch → none
→ activation dispersed

Scratch → Loc
→ local / boundary focused activation

Limitation
일부 correct Scratch
→ near-zero Grad-CAM

따라서 Grad-CAM은
모델의 spatial attention을 이해하는 데 유용했지만
모든 sample에서 안정적으로 해석 가능한 것은 아니었다.
5. Final Experiment Comparison
Model	Accuracy	Balanced Accuracy	Macro-F1
Baseline CNN	0.9663	0.7851	0.7745
Full Weighted CNN	0.9330	0.8356	0.7534
Tempered CNN	0.9670	0.7748	0.7861
Tempered + Augmentation	0.9660	0.8133	0.7655


6. Final Model
본 프로젝트의 대표 모델은
Tempered CNN
으로 선택하였다.
선택 이유:
Highest Accuracy
+
Highest Macro-F1
+
Improved Scratch Recall
+
Improved Loc Recall

Test performance:
Accuracy          : 96.70%
Balanced Accuracy : 77.48%
Macro-F1          : 78.61%

7. Key Findings
1. Accuracy Alone Was Misleading
Baseline CNN은 96.63%의 높은 Accuracy를 기록하였다.
하지만 Scratch recall은 단 3.95%였다.
따라서 심한 class imbalance 환경에서는
Accuracy만으로 모델 성능을 평가하면 안 된다.
2. Full Class Weight Improved Minority Recall
Scratch recall:
3.95%
→
34.46%

로 크게 개선되었다.
하지만:
Accuracy
96.63%
→
93.30%

Macro-F1도 감소하였다.
3. Tempered Weight Produced the Best Overall Trade-off
sqrt class weighting을 적용한 결과:
Accuracy  : 96.70%
Macro-F1  : 78.61%

로 가장 좋은 전체 성능을 기록하였다.
4. Augmentation Improved Recall but Not Overall F1
Rotation / Flip augmentation은
Scratch recall과 Balanced Accuracy를 개선하였다.
하지만 Macro-F1은 감소하였다.
따라서 augmentation이 모든 wafer defect class에
동일하게 유효하지 않음을 확인하였다.
5. Scratch Was the Most Difficult Defect Class
Tempered CNN 기준:
Scratch Test samples : 177
Correct              : 35
Wrong                : 142

오분류:
none     : 80
Loc      : 58
Edge-Loc : 4

Scratch와 normal / local pattern 간의
구분이 모델의 주요 failure mode 중 하나였다.
6. Grad-CAM Was Useful but Not Universally Reliable
Center에서는 의미 있는 attention map이 확인되었다.
하지만 일부 Scratch에서는
zero-CAM 또는 boundary-focused attention이 나타났다.
따라서 Grad-CAM 역시
모델 설명의 절대적인 근거가 아니라
보조적 interpretability tool로 사용해야 한다.
8. Limitations
본 프로젝트에는 다음 한계가 있다.
Dataset Imbalance
Near-full은 전체 labeled data에서 149개뿐이다.
Test set에서는 15개만 존재하기 때문에
해당 class의 성능은 통계적으로 불안정할 수 있다.
Dataset Representativeness
WM-811K의 class distribution은
해당 공개 데이터셋의 분포이다.
이를 실제 반도체 Fab의 불량 발생률이라고
해석해서는 안 된다.
Root Cause Interpretation
Wafer pattern classification은
공정 이상 분석을 지원할 수 있지만,
pattern
=
physical root cause

라고 단정할 수는 없다.
동일한 pattern도 여러 공정 원인에서 발생할 수 있다.
Augmentation Assumption
본 프로젝트에서는
90° rotation과 flip 이후에도
class label이 유지된다고 가정하였다.
그러나 실제 제조 환경에서는
wafer orientation 자체가
공정 원인 분석에 중요한 정보일 가능성이 있다.
Grad-CAM Limitation
Grad-CAM은
모델의 spatial activation을 보여줄 뿐
model reasoning
physical cause
process mechanism

을 직접 설명하지 않는다.
일부 correctly classified sample에서는
near-zero CAM도 확인되었다.
9. Current Project Pipeline
WM-811K
    ↓
Data Loading
    ↓
Label Cleaning
    ↓
172,950 Labeled Wafers
    ↓
EDA
    ↓
64×64 Nearest Resize
    ↓
Lot-aware Split
    ↓
Train / Validation / Test
    ↓
Simple CNN Baseline
    ↓
Class Weight Experiment
    ↓
Tempered Weight Experiment
    ↓
Augmentation Experiment
    ↓
Model Comparison
    ↓
Grad-CAM
    ↓
Error Analysis

10. Next Steps
향후 다음 방향으로 프로젝트를 확장할 계획이다.
Error Analysis
- Scratch misclassification analysis
- Class-wise confusion analysis
- difficult wafer visualization
Explainability
- Grad-CAM sample expansion
- correct vs incorrect prediction comparison
- Grad-CAM++ comparison
Quality Analysis
- defect class Pareto chart
- class distribution analysis
- candidate process issue discussion
단, 공정 원인은 추정 가능한 후보 원인 수준으로만 제시할 예정이다.
Deployment
Streamlit 기반 demo를 구현할 수 있다.
예:
Upload wafer map
      ↓
Model Prediction
      ↓
Predicted Class
Confidence
Grad-CAM

형태의 간단한 inference interface를 목표로 한다.
11. Repository Structure
현재 repository는 다음 형태로 구성한다.
wafer-map-defect-classification/
│
├── README.md
│
├── notebooks/
│   └── 01_wafer_project.ipynb
│
├── results/
│
└── .gitignore

프로젝트 완료 후 필요에 따라 다음과 같이 정리할 예정이다.
wafer-map-defect-classification/
│
├── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_gradcam.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── metrics.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── gradcam.png
│   └── model_comparison.csv
│
└── .gitignore

12. Data Availability
본 repository에는 원본 WM-811K 데이터 파일을 포함하지 않는다.
다음 파일은 GitHub에 업로드하지 않는다.
LSWMD.pkl
train_lot_split.pkl
val_lot_split.pkl
test_lot_split.pkl
*.pt

데이터는 WM-811K 공개 데이터셋에서 별도로 다운로드하여 사용한다.
13. Project Status
현재 진행 상태:
Dataset inspection        ✅
EDA                       ✅
Preprocessing             ✅
Lot-aware split           ✅
Baseline CNN              ✅
Class weighting           ✅
Tempered weighting        ✅
Augmentation              ✅
Grad-CAM                  ✅
Scratch error analysis    ✅

Detailed error analysis   🔄
Pareto analysis           ⏳
Streamlit demo            ⏳
Final README cleanup      ⏳


그리고 README를 더 포트폴리오답게 만들려면 최종적으로는 Day별 일지를 그대로 다 보여주기보다, 위 내용을 아래 4개 층으로 압축해서 배치하는 게 가장 좋아.

**첫 화면에는 `Overview → 핵심 결과표 → 대표 Grad-CAM 이미지`를 먼저 보여주고**, 그 아래에 `Methodology → Day별 상세 기록 → Limitations → Next Steps`를 두는 구조가 좋아. 채용 담당자는 보통 README를 처음부터 끝까지 읽지 않아서, 처음 30초 안에 “무슨 문제를 풀었고, 왜 특별한가”가 보여야 하거든.

특히 이 프로젝트에서 강조할 세 문장은 이걸로 잡으면 된다.

> **1. 동일 생산 lot의 데이터 누수를 막기 위해 lot-aware split을 적용했다.**

> **2. 높은 Accuracy에 가려진 minority class 성능 문제를 발견하고, Full Weight → Tempered Weight → Augmentation 순으로 불균형 대응 전략을 비교했다.**

> **3. Grad-CAM과 Scratch 오분류 분석을 통해 모델 성능뿐 아니라 공간적 판단 근거와 failure mode까지 분석했다.**

이 세 문장이 지금 프로젝트의 가장 강한 스토리야.
