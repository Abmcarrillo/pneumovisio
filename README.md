# Pneumonia Detection Using Chest X-rays - Pneumovisio

This project demonstrates a machine learning pipeline to detect patterns associated with pneumonia from chest X-ray images. The pipeline involves preprocessing, feature extraction, and classification using a Support Vector Machine (SVM).

## Overview
Pneumonia is a severe respiratory condition that can be detected using chest X-rays. This repository implements an automated system for detecting pneumonia using advanced image processing and machine learning techniques.

## Dataset
The dataset used for this project was sourced from Kaggle's [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). It includes X-ray images classified into two categories:

- **Normal**: Chest X-rays of healthy patients.
- **Pneumonia**: Chest X-rays of patients diagnosed with pneumonia.

## Methodology

### 1. Preprocessing
- **Image Resizing**: All images were resized to 128x128 pixels.
- **Grayscale Conversion**: Simplified images to a single intensity channel.
- **Gaussian Filtering**: Reduced noise for better feature extraction.

### 2. Feature Extraction
- **HOG (Histogram of Oriented Gradients)**: Captures the structure and texture of X-ray images.

### 3. Normalization
- **Z-score Normalization**: Ensures all features contribute equally to the model by scaling them to have a mean of 0 and standard deviation of 1.

### 4. Classification
- **SVM (Support Vector Machine)**: A linear kernel was used to separate the two classes with the maximum margin.
- Hyperparameter tuning was conducted to optimize the regularization parameter `C`.

## Results
The model was evaluated using the following metrics:

- **Accuracy**: Proportion of correctly classified images.
- **Precision**: Proportion of true positive pneumonia cases among all predicted pneumonia cases.
- **Recall**: Proportion of true positive pneumonia cases among all actual pneumonia cases.
- **F1-Score**: Harmonic mean of precision and recall.

## Visualizations
1. **Feature Normalization**:
   A comparison of original and normalized features shows how normalization scales features for equal contribution.
2. **SVM Decision Boundary**:
   A graphical representation of how the SVM separates classes using a linear hyperplane.

## How to Run

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `opencv-python`, `joblib`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/abmcarrillo/pneumovisio.git
   cd pneumonia-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set dataset paths in the script:
   Modify the `ruta_normal` and `ruta_neumonia` variables in every script to point to the `NORMAL` and `PNEUMONIA` image folders.
4. Run the script:
   ```bash
   pneumovisio_software.py or pneumovisio_training_machinelear_plus_ia.py
   ```


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset: [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- Libraries: OpenCV, scikit-learn, Matplotlib

## Contact
For questions or collaboration, feel free to reach out:
- **Email**: carrillobricenoabraham@gmail.com
- **GitHub**: [abmcarrillo](https://github.com/abmcarrillo)
