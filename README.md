# Intel_Image_Classification


## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify natural scene images into **6 classes** using the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

The goal was to practice:
- Data preprocessing & augmentation  
- Designing and training a CNN  
- Evaluating performance with metrics and visualizations  

---

## 📂 Dataset
The dataset contains **25,000+ images** of natural scenes across **6 classes**:

- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street  

**Dataset split:**
- `seg_train/` → Training images (split into train/validation using `ImageDataGenerator`)  
- `seg_test/` → Test images (used for final evaluation)

---

## 🧹 Data Preprocessing
- All images resized to **150×150**
- Pixel values normalized (`rescale=1./255`)
- Applied data augmentation:
  - Rotation (±20°)
  - Width & height shift
  - Shear, zoom
  - Horizontal flip

---

## 🧠 CNN Model Architecture
| Layer | Filters/Units | Kernel/Pool | Activation |
|------|---------------|------------|-----------|
| Conv2D + MaxPooling | 32 | 3×3, 2×2 | ReLU |
| Conv2D + MaxPooling | 64 | 3×3, 2×2 | ReLU |
| Conv2D + MaxPooling | 128 | 3×3, 2×2 | ReLU |
| Flatten | - | - | - |
| Dense | 128 | - | ReLU |
| Dropout | 0.3 | - | - |
| Dense (Output) | 6 | - | Softmax |

**Loss Function:** `categorical_crossentropy`  
**Optimizer:** `adam`  
**Metrics:** `accuracy`  

---
<img width="990" height="374" alt="image" src="https://github.com/user-attachments/assets/26c943f5-66c7-46d5-b683-146b275cba95" />

## 🛠 Tech Stack
- Python 3.x  
- TensorFlow / Keras  
- NumPy, Matplotlib, Seaborn  
- Scikit-learn (metrics)

