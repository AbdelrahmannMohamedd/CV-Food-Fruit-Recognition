# Food/Fruit Recognition and Calorie Estimation

Computer Vision project (2026, SC & AI) that identifies whether an image contains **food or fruit**, classifies the specific **food/fruit category**, and estimates **calorie content**. Built with TensorFlow/Keras and transfer learning on MobileNetV2.

## Project Overview

The pipeline is split into stages, following a two-stage identification approach:

1. **Stage 1 — Food vs. Fruit Classification (Part A):** Binary classifier (MobileNetV2 backbone) that determines whether an input image is food or fruit.
2. **Stage 2a — Food Recognition (Part B):** One/few-shot learning via a Siamese network (triplet loss) to recognize food categories, including unseen classes at test time.
3. **Stage 2b — Fruit Classification (Part C):** Multi-class classifier (MobileNetV2 backbone) that predicts the specific fruit category.

## Repository Structure

```
.
├── Part_A.py                     # Stage 1: Food vs. Fruit binary classifier
├── Part_B.py                     # Stage 2a: Siamese network for food recognition
├── Part_C.py                     # Stage 2b: Fruit type classifier
├── models/
│   ├── model_part_a_binary.keras
│   ├── model_part_b_encoder.keras
│   └── model_part_c_fruit.keras
├── ProjectDescription2026_SC_AI.pdf
└── README.md
```

> **Note:** The dataset (`Project Data/`) is not included in this repository due to size.

## Models

| Part | Task | Architecture | Output |
|------|------|--------------|--------|
| A | Food vs. Fruit (binary) | MobileNetV2 (frozen) + Dense(1, sigmoid) | `model_part_a_binary.keras` |
| B | Food recognition (few-shot) | MobileNetV2 (partially fine-tuned) + Siamese/Triplet loss | `model_part_b_encoder.keras` |
| C | Fruit classification (multi-class) | MobileNetV2 (frozen) + Dense(N, softmax) | `model_part_c_fruit.keras` |

## Dataset

The dataset follows this structure and should be placed at `Project Data/` in the repo root before training:

```
Project Data/
├── Food/
│   ├── Train/<category>/*.jpg
│   └── Validation/<category>/*.jpg
└── Fruit/
    ├── Train/<category>/Images/*.jpg
    └── Validation/<category>/Images/*.jpg
```

## Setup

```bash
pip install tensorflow numpy
```

## Usage

Train each stage independently:

```bash
python Part_A.py   # Food vs. Fruit classifier
python Part_B.py   # Siamese food-recognition encoder
python Part_C.py   # Fruit type classifier
```

Each script saves its trained model to the project root (e.g. `model_part_a_binary.keras`).

## Status / Roadmap

- [x] Part A — Food/Fruit binary classification
- [x] Part B — Food recognition (Siamese, one/few-shot)
- [x] Part C — Fruit type classification
- [ ] Part D — Fruit binary segmentation
- [ ] Part E — Fruit multi-class segmentation (31 classes)
- [ ] Individual test scripts (Parts A–E)
- [ ] Integrated test pipeline (classification + calorie estimation + segmentation)
- [ ] Final report

## License

Academic project — for coursework purposes.
