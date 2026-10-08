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
