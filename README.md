# Human Face Recognition Using Deep Learning

A Google Colab-based academic project that recognizes known celebrity faces in images using face detection, face embeddings, and a neural-network classifier. The notebook also includes an unknown-person rejection step and an optional CSV-based attendance log.

## Project Overview

Traditional image classification can be sensitive to changes in pose, lighting, and face position. This project uses a face-processing pipeline to locate and align a face, convert it into a numerical representation (embedding), and classify that representation as one of the known identities.

## Main Features

- Downloads a celebrity-face dataset from Kaggle.
- Organizes images into identity-based folders.
- Detects faces using MTCNN and uses eye keypoints to help align them.
- Resizes detected faces to 160 × 160 pixels.
- Extracts facial embeddings using FaceNet (`keras-facenet`).
- Trains a TensorFlow/Keras dense neural-network classifier on the embeddings.
- Uses confidence and cosine-distance thresholds to label uncertain matches as `Unknown`.
- Evaluates predictions with accuracy, precision, recall, F1-score, a classification report, and a confusion matrix.
- Saves model and preprocessing artifacts.
- Supports testing an uploaded image and optionally records recognized identities in a CSV attendance file.
- Visualizes embedding patterns using PCA.

## Technology Stack

- **Language:** Python
- **Notebook environment:** Google Colab
- **Deep learning:** TensorFlow / Keras
- **Face detection:** MTCNN
- **Face embeddings:** FaceNet (`keras-facenet`)
- **Image processing:** OpenCV
- **Data handling:** NumPy, Pandas
- **Machine-learning utilities:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Dataset source:** Kaggle — `bhaveshmittal/celebrity-face-recognition-dataset`
- **Storage:** Local/Colab folders for images and model artifacts; Pandas CSV for attendance (no SQL/NoSQL database is implemented in this notebook)

## End-to-End Workflow

1. **Dataset acquisition:** Upload the Kaggle API token (`kaggle.json`) in Colab and download the celebrity-face dataset.
2. **Folder preparation:** Extract images and organize them into folders by identity. Each folder name is treated as a class label.
3. **Exploratory analysis:** Inspect class distribution and sample images (the corresponding data-preparation definitions must be present in the notebook).
4. **Dataset split:** Split records into training, validation, and test sets using stratification to preserve class proportions.
5. **Face detection:** MTCNN locates faces and facial keypoints in each image.
6. **Face alignment and normalization:** Use eye positions to estimate rotation, crop the detected face, convert color format, and resize it to 160 × 160.
7. **Embedding extraction:** FaceNet converts each aligned face into a numerical feature vector representing facial characteristics.
8. **Classifier training:** Standardize the embedding features, encode identity labels, and train a dense neural network with ReLU layers and a Softmax output. Validation monitoring, checkpointing, learning-rate reduction, and early stopping are used.
9. **Evaluation:** Evaluate the selected model on held-out test data and inspect classification metrics and the confusion matrix.
10. **Prediction:** For an uploaded image, detect faces, extract embeddings, scale them with the saved scaler, and obtain class probabilities from the trained classifier.
11. **Unknown rejection:** Compare the top class probability and cosine distance to the corresponding class centroid with configured thresholds. If the match fails either check, return `Unknown`.
12. **Optional attendance record:** For a recognized identity, append the name, date, and time to `models/attendance.csv`, avoiding a second entry for the same name on the same date.

## Architecture

```text
Kaggle Celebrity Image Dataset
              |
              v
   Image folders / class labels
              |
              v
       Train / Val / Test split
              |
              v
       MTCNN face detection
              |
              v
   Face alignment + resize (160x160)
              |
              v
       FaceNet embedding vector
              |
              v
  StandardScaler + label encoding
              |
              v
    Dense TensorFlow/Keras model
              |
       Evaluation + saved artifacts
              |
              v
        Input / uploaded image
              |
              v
 MTCNN -> alignment -> FaceNet embedding
              |
              v
       Scaler -> classifier scores
              |
              v
 Confidence + centroid-distance checks
              |
        +-----+------+
        |            |
   Known identity  "Unknown"
        |
        v
 Optional CSV attendance log
```

## Data and Model Storage

This notebook does **not** use a relational database such as MySQL or a NoSQL database such as MongoDB.

- `dataset/raw/`: extracted source dataset images.
- `dataset/structured/`: images organized by identity/class folder.
- `models/best_face_classifier.keras`: saved Keras classifier checkpoint.
- `models/label_encoder.pkl`: mapping between class names and numeric labels.
- `models/scaler.pkl`: fitted feature scaler used before classification.
- `models/class_centroids.npy`: class-centroid embedding vectors used for distance-based rejection.
- `models/class_names.npy`: saved class names.
- `models/attendance.csv`: optional attendance entries containing name, date, and time.

These files are written to the notebook's working directory (or `/content/Human-Face-Recognition` in Colab). Colab storage may be temporary unless files are downloaded or saved to persistent storage.

## Run the Project

### Requirements

The notebook is designed to run in **Google Colab** and downloads its dataset through the Kaggle API.

1. Upload `22121018.ipynb` to Google Colab.
2. In Kaggle account settings, create an API token and download `kaggle.json`.
3. Ensure the Kaggle account has access to the dataset referenced in the notebook.
4. Run the setup and dependency-installation cells.
5. Upload `kaggle.json` when prompted. Do not commit this token to GitHub.
6. Run the data preparation, split, embedding extraction, training, and evaluation cells in order.
7. Upload a test image in the inference cell to view detected faces and predicted identities.

### Dependencies

The notebook installs packages including:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow scikit-learn joblib tqdm mtcnn keras-facenet deepface requests
```

## Configuration

The notebook defines these initial settings:

| Setting                         |   Default |
| ------------------------------- | --------: |
| Face input size                 | 160 × 160 |
| Minimum images per class        |        20 |
| Maximum classes                 |        17 |
| Test split                      |       15% |
| Validation split                |       15% |
| Batch size                      |        32 |
| Maximum training epochs         |        10 |
| Unknown distance threshold      |      0.40 |
| Classifier confidence threshold |      0.60 |

Thresholds are project configuration values, not guarantees of reliable recognition. They should be validated on a representative held-out dataset before drawing conclusions.

## Evaluation

The notebook calculates and displays:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Per-class classification report
- Confusion matrix
- Training/validation accuracy and loss curves
- PCA visualization of a sample of face embeddings

## Current Limitations / Before Publishing

- The PDF supplied with the project appears to contain image-only pages and no extractable text, so this README is based mainly on the notebook source.
- Verify the notebook runs from a fresh Colab runtime. In the uploaded notebook, the exploratory-analysis cell references `df`, `plot_distribution`, and `show_samples`; ensure their dataset-building/function definitions are included and executed before that cell.
- Confirm that the selected checkpoint, scaler, label encoder, centroids, and class names are saved and loaded consistently.
- Recognition quality depends on dataset quality, class balance, image conditions, and threshold selection.
- The notebook records attendance to CSV, not to a database server.
- Do not publish `kaggle.json`, private images, or sensitive attendance records in the repository.
- Obtain consent and follow applicable privacy rules when processing identifiable face images.

## Author

Hrutik Tukaram Wakale, Government College of Engineering And Research Avasari Khurd.
