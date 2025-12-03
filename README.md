# Diabetic Retinopathy Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements a **custom ResNet-based deep learning architecture** for automated multi-class classification of diabetic retinopathy severity from retinal fundus images. Diabetic retinopathy is a leading cause of preventable blindness worldwide, affecting approximately 1 in 3 diabetic patients. Early detection through automated screening can significantly improve patient outcomes.

**Key Achievement:** Achieved **83% classification accuracy** across 5 severity classes using a custom-built residual neural network trained on the Kaggle Diabetic Retinopathy Detection dataset.

---

## 🎯 Problem Statement

Diabetic retinopathy progresses through distinct severity stages, requiring different clinical interventions:

| Class | Severity | Clinical Action |
|-------|----------|-----------------|
| 0 | No DR | Annual screening |
| 1 | Mild | 6-12 month follow-up |
| 2 | Moderate | 3-6 month follow-up |
| 3 | Severe | Referral to specialist |
| 4 | Proliferative DR | Urgent treatment required |

Manual screening by ophthalmologists is time-consuming and subject to variability. This project develops an **automated classification pipeline** that can assist clinicians in prioritizing high-risk patients.

---

## 🔬 Technical Approach

### Architecture: Custom ResNet Implementation

Built a **custom residual neural network from scratch** (not using pre-trained weights) to understand the fundamental principles of skip connections and deep network training:

```
Input (256×256×3)
    ↓
Initial Conv Block (7×7, stride=2) + BatchNorm + ReLU + MaxPool
    ↓
ResBlock Stage 2: [64, 64, 256] filters
    ↓
ResBlock Stage 3: [128, 128, 512] filters
    ↓
ResBlock Stage 4: [256, 256, 1024] filters
    ↓
Average Pooling → Flatten → Dense(5, softmax)
```

**Key Design Decisions:**
- **Custom ResBlock Implementation:** Each block contains convolutional + identity sub-blocks with skip connections
- **Glorot Uniform Initialization:** For stable gradient flow during training
- **Batch Normalization:** Applied after each convolution for training stability
- **Progressive Feature Scaling:** Filter sizes double at each stage following ResNet conventions

### Data Pipeline & Augmentation

Implemented a robust data pipeline using TensorFlow's `ImageDataGenerator`:

- **Training Set:** 80% of data with real-time augmentation
  - Rescaling (0-1 normalization)
  - Shear transformation (0.2 range)
  - Validation split (15% held out)
- **Test Set:** 20% of data with only normalization
- **Batch Size:** 32 images
- **Image Resolution:** 256×256 RGB

### Training Configuration

- **Optimizer:** Adam with default learning rate
- **Loss Function:** Categorical Cross-Entropy
- **Callbacks:**
  - `EarlyStopping`: Patience=15, monitoring validation loss
  - `ModelCheckpoint`: Saving best weights based on validation performance

---

## 📊 Results & Evaluation

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 83% |
| **Training Convergence** | Stable loss reduction |
| **Classes** | 5-way classification |

### Evaluation Methodology

- **Confusion Matrix Analysis:** Visualized per-class performance to identify challenging classifications
- **Classification Report:** Precision, recall, and F1-score for each severity level
- **Visual Inspection:** Random sample predictions compared against ground truth labels

---

## 🛠️ Technical Components

| Category | Technologies & Techniques |
|----------|--------------------------|
| **Deep Learning Frameworks** | TensorFlow 2.x, Keras |
| **Neural Network Architecture** | Custom ResNet, Skip Connections, Batch Normalization |
| **Computer Vision** | Image Classification, Data Augmentation, Transfer Learning Concepts |
| **Data Processing** | Pandas, NumPy, PIL, OpenCV |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML Best Practices** | Train/Val/Test Split, Early Stopping, Model Checkpointing |
| **Evaluation Metrics** | Confusion Matrix, Classification Report, Accuracy Analysis |

---

## 📁 Repository Structure

```
├── diabetic_retinopathy_classification.ipynb  # Main analysis notebook
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies
├── weights.hdf5                                # Trained model weights
├── retina_weights.hdf5                         # Best performing weights
├── train.csv                                   # Image labels mapping
└── train/                                      # Training images
    ├── No_DR/                                  # Class 0: No diabetic retinopathy
    ├── Mild/                                   # Class 1: Mild severity
    ├── Moderate/                               # Class 2: Moderate severity
    ├── Severe/                                 # Class 3: Severe severity
    └── Proliferate_DR/                         # Class 4: Proliferative DR
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
CUDA-compatible GPU (recommended)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/sahilpbhatt1/Diabetic-Retinopathy-Diagnosis-using-AI.git
cd Diabetic-Retinopathy-Diagnosis-using-AI

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook diabetic_retinopathy_classification.ipynb
```

### Running the Model

1. **Training:** Execute cells sequentially to train the model from scratch
2. **Inference:** Load pre-trained weights from `retina_weights.hdf5` for immediate predictions
3. **Evaluation:** Run evaluation cells to generate confusion matrix and classification report

---

## 📚 Dataset

**Source:** [Kaggle Diabetic Retinopathy Detection Challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection)

The dataset contains high-resolution retinal fundus photographs taken under various imaging conditions. Images are labeled by trained clinicians on a 0-4 scale representing severity of diabetic retinopathy.

---

## 🔮 Future Improvements

- [ ] Implement attention mechanisms for interpretable predictions
- [ ] Explore transformer-based architectures (Vision Transformers)
- [ ] Add Grad-CAM visualizations for model interpretability
- [ ] Develop REST API for model deployment
- [ ] Experiment with class balancing techniques for imbalanced data

---

## 📖 References

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
2. Google AI & Aravind Eye Hospital collaboration: [VentureBeat Article](https://venturebeat.com/2019/02/25/google-works-with-aravind-eye-hospital-to-deploy-ai-that-can-detect-eye-disease/)

---

## 👤 Author

**Sahil Bhatt**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
