# 🌿 Plant Leaf Disease Detection using CNN

Deep Learning based Computer Vision system for automatic detection of plant leaf diseases using Convolutional Neural Networks (CNN).

---

## 📌 Project Overview

This project builds a Convolutional Neural Network (CNN) to classify plant leaf diseases from images.  
The model is trained on RGB leaf images resized to **128×128** resolution.

Due to GitHub file size limitations (25MB), the trained model and dataset are provided via external links.

---

## 🧠 Model Architecture

The CNN model is built using TensorFlow/Keras Sequential API.

### 🔹 Architecture Summary:

- Conv2D (32 filters) + BatchNorm
- Conv2D (32 filters)
- MaxPooling + Dropout
- Conv2D (64 filters) + BatchNorm
- Conv2D (64 filters)
- MaxPooling + Dropout
- Conv2D (128 filters) + BatchNorm
- Conv2D (128 filters)
- MaxPooling + Dropout
- Fully Connected (512 units)
- Dropout (0.5)
- Output Layer (Softmax)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
