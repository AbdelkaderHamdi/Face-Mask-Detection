# Face Mask Detection using CNN

This project implements a face mask detection system using a Convolutional Neural Network (CNN) with TensorFlow and Keras. It includes scripts for model training and real-time evaluation via a webcam.

## Project Overview
This project aims to classify whether a person is wearing a face mask, wearing it incorrectly, or not wearing one at all. A CNN model is trained to perform this classification in real-time.

## Features
-   **Data Preparation**: Automatic splitting of the dataset into training, validation, and test sets.
-   **Data Augmentation**: Utilizes `ImageDataGenerator` to enhance model robustness.
-   **CNN Model**: Builds and trains a sequential CNN for image classification.
-   **Model Evaluation**: Calculates precision, recall, and F1-score on the test set.
-   **Real-time Detection**: Applies the trained model for live face mask detection using a webcam.

## Installation
To set up and run this project, follow these steps:

1.  **Clone the repository** (or ensure you have `Face_Mask_Detection_Using_CNN.py` and `Evaluate_real_time.py`).

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Model Training
The `Face_Mask_Detection_Using_CNN.py` script handles dataset download, splitting, CNN training, and performance evaluation. To run it:

```bash
python Face_Mask_Detection_Using_CNN.py
```

This script will:
-   Download the `vijaykumar1799/face-mask-detection` dataset from KaggleHub.
-   Create a `dataset_split` folder structure for training, validation, and test sets.
-   Train the CNN model.
-   Display a classification report.
-   Save the trained model as `face_mask_detection_model.keras` in the `models` directory.

### Real-time Detection
The `Evaluate_real_time.py` script uses the trained model for live face mask detection via your webcam. Ensure `face_mask_detection_model.keras` is in the `models` folder before running.

```bash
python Evaluate_real_time.py
```

This script will:
-   Load the `face_mask_detection_model.keras`.
-   Open your webcam feed.
-   Detect faces (using a predefined Region of Interest) and predict mask usage.
-   Display real-time results with colored bounding boxes and labels.
-   Press 'q' to exit the real-time detection.

## Dataset
The project uses the **Face Mask Detection** dataset by `vijaykumar1799` from KaggleHub [1]. It includes images categorized as:
-   `mask_weared_incorrect`
-   `with_mask`
-   `without_mask`

The `Face_Mask_Detection_Using_CNN.py` script automatically downloads and prepares this dataset.

## Model Architecture
The CNN model is a sequential architecture comprising:
-   Multiple convolutional blocks (Conv2D + MaxPooling2D) for feature extraction.
-   A `Flatten` layer to convert 2D features to 1D.
-   `Dense` layers for classification.
-   A `Dropout` layer to prevent overfitting.

The model is compiled with the `adam` optimizer, `categorical_crossentropy` loss function, and `accuracy` metric.


## References
KaggleHub. *Face Mask Detection Dataset*. Retrieved from [https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)

