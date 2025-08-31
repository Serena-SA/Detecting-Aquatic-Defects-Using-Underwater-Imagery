# Aquatic Defect Detection (EfficientNet-B2, PyTorch)

This repository contains a modular PyTorch pipeline to detect and classify aquatic defects (biofouling, holes, vegetation) from underwater imagery using **EfficientNet-B2**.
It's worth noting that it's easy to change the model choice to any existing state-of-the-art image classification models.
---

## Tournament Achievement
This code was submitted to the **[Machine Vision Innovation Tournament](https://taimurhassan.github.io/mvi_icip/)**, held as part of the **IEEE's International Conference of Image Processing (ICIP)**, where it won **2nd Place**.

## File Overview

- `load_data.py` – Prepares datasets (train/val) with transforms and handles class imbalance.  
- `loss.py` – Contains loss functions (CrossEntropy, hinge).  
- `train_iter.py` – Core training loop with metrics (loss, accuracy, recall, precision, IoU, F1).  
- `train_model.py` – Main script: loads data, builds EfficientNet-B2, trains, plots metrics, saves weights.  
- `model_converter.py` – Converts saved state_dict to a full model (architecture + weights).  
- `Testing_the_model.py` – Evaluates on test data, prints metrics, and generates a confusion matrix.  
- `test_cuda.py` – Quick CUDA/GPU availability check.  

---

## Workflow

### 1. Check GPU
In bash, or run:
```bash
python test_cuda.py
```

### 2. Train model
```bash
python train_model.py
```

### 3. Convert model
This converts the .pth weights created in train_model into the full architecture of choice
through the existing model that was trained on ImageNet.
```bash
python model_converter.py
```

### 4. Evaluate
This prints metrics (accuracy, precision, recall, F1, IoU) and generates a confusion matrix.
```bash
python Testing_the_model.py
```
---

## Citation

If you use this work, please cite both the dataset authors and the original base code:

- **Dataset:** Underwater imagery dataset described in [this ScienceDirect article](https://www.sciencedirect.com/science/article/abs/pii/S0957417425004427?via%3Dihub). Please cite the authors if you use the dataset.  
- **Base code reference:** Adapted from [Nivitus/PyTorch_Image_Classifier](https://github.com/Nivitus/PyTorch_Image_Classifier).  

---
