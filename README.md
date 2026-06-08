# PILL: Physics-Informed Learning for Bearing Fault Diagnosis

A multimodal deep learning system for intelligent bearing fault diagnosis using vibration signals, physics-based features, and machine operating conditions.

## Overview

PILL combines three specialized neural networks and an attention-based fusion mechanism to detect bearing faults from industrial machinery data.

The system integrates:

* Spectrogram-based vibration analysis
* Physics-informed fault features
* Operating condition metadata

Instead of relying on a single source of information, PILL adaptively learns which modality is most informative for each sample and explains its decisions through attention weights.

## Key Features

* Multi-modal fault diagnosis
* Physics-informed feature engineering
* Attention-based model fusion
* Explainable predictions
* Out-of-distribution detection
* Real-time inference support
* Industrial predictive maintenance application

## Architecture

```text
Raw Bearing Signal
        ↓
Preprocessing
        ↓
Feature Extraction

   ↙      ↓      ↘

 CNN   Physics   Metadata
Branch  Branch   Branch

   ↘      ↓      ↙

Attention-Based Fusion
        ↓
Fault Classification
        ↓
Explainable Predictions
```
## Dataset

* SCA Bearing Dataset
* Raw vibration signals sampled at 12 kHz
* Multiple bearing types and operating conditions
* Binary classification:

  * Healthy
  * Faulty

## Technology Stack

### Machine Learning

* PyTorch
* NumPy
* SciPy
* Librosa

### Backend

* FastAPI

### Frontend

* React
* TypeScript
* Tailwind CSS

## Project Structure

```text
src/
├── data/
├── models/
├── training/
├── losses/
├── utils/
└── main.py

frontend/
├── src/
└── ...
```

## Results

| Model              | Purpose                         |
| ------------------ | ------------------------------- |
| Spectrogram Branch | Frequency-domain fault patterns |
| Physics Branch     | Bearing fault signatures        |
| Metadata Branch    | Operating condition context     |
| Fusion Model       | Adaptive ensemble prediction    |

The fusion model combines predictions from all three branches using learned attention weights to improve robustness and interpretability.

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python src/train_branch_models.py
python src/train_fusion.py
```

### Run Inference

```bash
python src/main.py
```

## Future Work

* Live monitoring dashboard
* Streaming sensor support
* Edge deployment
* Model serving with Docker

## License

MIT License
