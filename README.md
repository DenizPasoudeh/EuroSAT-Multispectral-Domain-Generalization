# EuroSAT Multispectral Domain Generalization Classification

This project implements a **multispectral domain generalization pipeline** for land-cover classification using the **EuroSAT multispectral Sentinel-2 dataset**.

The goal is to classify satellite images into land-cover categories while improving robustness against unseen domain shifts such as spectral changes, brightness/contrast variations, noise, and band degradation.

Unlike standard RGB image classification, this project uses **13 Sentinel-2 spectral bands**, making it suitable for remote sensing and multispectral image analysis.

---

## Project Overview

This project compares three models:

1. **RGB CNN Baseline**
   - Uses only RGB bands: B04, B03, B02.

2. **Multispectral CNN Baseline**
   - Uses all 13 Sentinel-2 bands.

3. **Multispectral Domain Generalization CNN**
   - Uses all 13 bands.
   - Trained with domain randomization techniques such as brightness/contrast shifts, spectral noise, and spectral scaling.

The main objective is to evaluate whether multispectral domain randomization improves performance on an **unseen target domain**.

---

## Dataset

The project uses the **EuroSAT multispectral dataset**, which contains Sentinel-2 satellite image patches.

Classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

---

## Dataset Visualization

### RGB Samples

![RGB Samples](assets/rgb_samples.png)

### False-Color Samples

![False Color Samples](assets/false_color_samples.png)

### Example of All 13 Spectral Bands

![All 13 Bands](assets/all_13_bands_sample.png)

---

### Domain Shift Examples

![Domain Shift Examples](assets/domain_shift_examples.png)

---

## Models

### 1. RGB CNN Baseline

This model uses only three Sentinel-2 bands:

- B04 as Red
- B03 as Green
- B02 as Blue

It represents a standard RGB image classification baseline.

### 2. Multispectral CNN Baseline

This model uses all 13 Sentinel-2 bands as input.

Input shape:

```text
64 × 64 × 13
```

### 3. Multispectral Domain Generalization CNN

This model also uses all 13 bands but is trained with domain randomization. During training, the model sees different augmented versions of multispectral images, helping it become more robust to unseen domain shifts.

---

## Results

The models were evaluated on:

1. Clean test set
2. Unseen target domain test set

| Model | Clean Accuracy | Target Accuracy | Accuracy Drop | Clean F1 Macro | Target F1 Macro |
|---|---:|---:|---:|---:|---:|
| RGB CNN Baseline | 0.5387 | 0.4500 | 0.0887 | 0.5349 | 0.4276 |
| Multispectral CNN Baseline | 0.4947 | 0.1633 | 0.3313 | 0.4811 | 0.1136 |
| Multispectral Domain Generalization CNN | **0.8713** | **0.6467** | 0.2247 | **0.8710** | **0.6092** |

The domain generalization model achieved the best performance on both the clean test set and the unseen target domain.

### Model Comparison

![Model Comparison](assets/model_comparison_accuracy.png)

---

## How to Run

The project is designed to run in Google Colab.

1. Open the notebook:

```text
notebooks/Multispectral-Domain-Generalization.ipynb
```

2. Run the cells in order.

3. The notebook will:
   - Download the EuroSAT multispectral dataset
   - Extract and read TIFF images
   - Visualize multispectral samples
   - Create synthetic domain shifts
   - Train three CNN models
   - Evaluate the models on clean and unseen target domains
   - Save metrics and visual outputs

---

## Requirements

The main dependencies are:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
tifffile
tqdm
```

Install them with:

```bash
pip install -r requirements.txt
```

---
