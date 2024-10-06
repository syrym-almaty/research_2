# Gender Detection from Audio Signals: Analyzing Pitch and Training a Classifier Using Labeled Data

---

## Introduction

**Objective**: The goal of this notebook is to perform gender detection from a given audio file in the Kazakh language. We will explore two primary methods:

1. **Pitch Analysis**: Utilizing fundamental frequency (F0) to determine gender based on acoustic properties.
2. **Machine Learning Classifier**: Training a classifier using labeled data and Mel-Frequency Cepstral Coefficients (MFCCs) as features.

**Significance**: Gender detection is a fundamental task in speech processing with applications in speaker recognition, conversational agents, and sociolinguistic studies. Understanding the underlying acoustic features that differentiate male and female voices enhances the development of more natural and adaptive speech systems.

---

## Table of Contents

1. [Data Loading and Visualization](#1)
2. [Pitch Analysis](#2)
   - [Fundamental Concepts](#2.1)
   - [Pitch Extraction](#2.2)
   - [Gender Classification Based on Pitch](#2.3)
3. [Formant Analysis (Optional)](#3)
4. [Feature Extraction with MFCCs](#4)
   - [Understanding MFCCs](#4.1)
   - [MFCC Extraction](#4.2)
5. [Training a Machine Learning Classifier](#5)
   - [Preparing Labeled Data](#5.1)
   - [Model Training and Evaluation](#5.2)
6. [Conclusion](#6)
7. [References](#7)

---

## 1. Data Loading and Visualization

First, we will load the audio file and visualize its waveform and spectrogram to understand its characteristics.

### Code

```python
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

# Load audio file
def load_audio(file_path):
    """
    Load an audio file and return the audio time series and sampling rate.

    Parameters:
    - file_path: str, path to the audio file.

    Returns:
    - y: np.ndarray, audio time series.
    - sr: int, sampling rate.
    """
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Visualize waveform
def plot_waveform(y, sr):
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Visualize spectrogram
def plot_spectrogram(y, sr):
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title('Spectrogram')
    plt.show()

# Load and plot
file_path = r'C:\Users\syrym\Downloads\research_2\audio.wav'
y, sr = load_audio(file_path)
plot_waveform(y, sr)
plot_spectrogram(y, sr)
```

### Explanation

- **Waveform**: Displays amplitude over time, providing a time-domain representation of the audio signal.
- **Spectrogram**: Shows frequency content over time, useful for observing the energy distribution across frequencies.

---

## 2. Pitch Analysis

### 2.1 Fundamental Concepts

- **Fundamental Frequency (F0)**: The lowest frequency of a periodic waveform, representing the pitch of the voice.
- **Gender Differences**:
  - **Male Voices**: Typically have an F0 ranging from 85 to 180 Hz.
  - **Female Voices**: Typically have an F0 ranging from 165 to 255 Hz.

### 2.2 Pitch Extraction

We will extract the pitch using the autocorrelation method provided by `librosa`.

#### Code

```python
def extract_pitch(y, sr, fmin=50, fmax=300):
    """
    Extract the fundamental frequency (F0) using the autocorrelation method.

    Parameters:
    - y: np.ndarray, audio time series.
    - sr: int, sampling rate.
    - fmin: float, minimum frequency to consider (Hz).
    - fmax: float, maximum frequency to consider (Hz).

    Returns:
    - pitches: np.ndarray, pitch (F0) values over time.
    - times: np.ndarray, time stamps.
    """
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=fmin, fmax=fmax)
    pitch_values = []
    times = []

    for i in range(pitches.shape[1]):
        pitch = pitches[:, i]
        mag = magnitudes[:, i]
        index = mag.argmax()
        pitch_freq = pitch[index]
        pitch_values.append(pitch_freq)
        times.append(i * (512 / sr))  # Assuming default hop_length=512

    pitches = np.array(pitch_values)
    times = np.array(times)
    return pitches, times

# Extract pitch
pitches, times = extract_pitch(y, sr)
```

#### Visualization

```python
def plot_pitch(pitches, times):
    plt.figure(figsize=(14, 4))
    plt.plot(times, pitches, color='g')
    plt.title('Pitch Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 500)
    plt.show()

plot_pitch(pitches, times)
```

### 2.3 Gender Classification Based on Pitch

We compute the average pitch and classify gender based on a threshold.

#### Code

```python
def classify_gender(pitches):
    """
    Classify gender based on average pitch.

    Parameters:
    - pitches: np.ndarray, array of pitch (F0) values.

    Returns:
    - str, predicted gender ('Male' or 'Female').
    """
    # Remove zero and NaN values
    pitches = pitches[(pitches > 0) & (~np.isnan(pitches))]
    avg_pitch = np.mean(pitches)
    print(f"Average Pitch: {avg_pitch:.2f} Hz")

    # Threshold for classification
    threshold = 165  # Hz

    if avg_pitch > threshold:
        return 'Female'
    else:
        return 'Male'

predicted_gender = classify_gender(pitches)
print(f"Predicted Gender: {predicted_gender}")
```

#### Explanation

- **Statistical Analysis**: We compute the mean of the non-zero pitch values.
- **Thresholding**: Based on literature, a threshold around 165 Hz separates male and female voices.

---

## 3. Formant Analysis (Optional)

*This section is optional and requires additional libraries and complex signal processing.*

---

## 4. Feature Extraction with MFCCs

### 4.1 Understanding MFCCs

- **MFCCs**: Mel-Frequency Cepstral Coefficients capture the short-term power spectrum of a sound.
- **Relevance**: MFCCs mimic the human auditory system and are effective features for speech and speaker recognition tasks.

### 4.2 MFCC Extraction

#### Code

```python
def extract_mfccs(y, sr, n_mfcc=13):
    """
    Extract MFCCs from the audio signal.

    Parameters:
    - y: np.ndarray, audio time series.
    - sr: int, sampling rate.
    - n_mfcc: int, number of MFCCs to extract.

    Returns:
    - mfccs: np.ndarray, MFCC feature matrix.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

mfccs = extract_mfccs(y, sr)
print(f"MFCCs shape: {mfccs.shape}")
```

#### Visualization

```python
def plot_mfccs(mfccs):
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.show()

plot_mfccs(mfccs)
```

#### Explanation

- **MFCC Matrix**: The result is a 2D array where rows represent MFCC coefficients, and columns represent frames over time.
- **Linear Algebra Perspective**: MFCCs are obtained by applying the Discrete Cosine Transform (DCT) to the log Mel-scaled power spectrum.

---

## 5. Training a Machine Learning Classifier

### 5.1 Preparing Labeled Data

To train a classifier, we need a dataset of audio samples with known gender labels.

#### Dataset Considerations

- **Sources**: Public speech datasets like **Mozilla Common Voice**, **VoxCeleb**, or any Kazakh language datasets.
- **Features**: We will use MFCCs as input features.
- **Labels**: Gender labels (0 for male, 1 for female).

#### Code

```python
# Placeholder function to load dataset
def load_dataset():
    """
    Load and preprocess the dataset.

    Returns:
    - X: np.ndarray, feature matrix.
    - y: np.ndarray, labels.
    """
    # For demonstration, let's simulate some data
    # In practice, load your actual dataset
    X = []
    y = []

    # Simulate data loading
    # Assume we have lists: audio_files (paths) and labels (0 or 1)
    audio_files = ['path_to_male_audio.wav', 'path_to_female_audio.wav']  # Replace with actual paths
    labels = [0, 1]

    for file_path, label in zip(audio_files, labels):
        y_i, sr_i = load_audio(file_path)
        mfccs_i = extract_mfccs(y_i, sr_i)
        # Average MFCCs over time
        mfccs_mean = np.mean(mfccs_i, axis=1)
        X.append(mfccs_mean)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y

# Load dataset
X, y = load_dataset()
print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

### 5.2 Model Training and Evaluation

We will train a logistic regression classifier.

#### Code

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data
def split_data(X, y):
    """
    Split the dataset into training and testing sets.

    Returns:
    - X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
def scale_features(X_train, X_test):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Returns:
    - X_train_scaled, X_test_scaled
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train model
def train_model(X_train, y_train):
    """
    Train a logistic regression classifier.

    Returns:
    - model: trained model
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.

    Returns:
    - None
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Run the pipeline
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
model = train_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)
```

#### Explanation

- **Standardization**: Important for models like logistic regression to perform optimally.
- **Evaluation Metrics**:
  - **Accuracy**: Overall correctness of the model.
  - **Confusion Matrix**: Shows true vs. predicted classifications.
  - **Classification Report**: Includes precision, recall, and F1-score.

---

## Using the Trained Model on Your Audio File

Now, we can use the trained model to predict the gender of the voice in your audio file.

#### Code

```python
# Extract MFCCs from your audio file
mfccs_your_audio = extract_mfccs(y, sr)
mfccs_your_audio_mean = np.mean(mfccs_your_audio, axis=1).reshape(1, -1)

# Scale features
mfccs_your_audio_scaled = scaler.transform(mfccs_your_audio_mean)

# Predict gender
gender_pred = model.predict(mfccs_your_audio_scaled)
gender_label = 'Female' if gender_pred[0] == 1 else 'Male'
print(f"The predicted gender is: {gender_label}")
```

#### Note

- Ensure that the scaler (`scaler`) and model (`model`) are defined in the current scope.
- Replace `extract_mfccs`, `load_audio`, and other functions with actual implementations if necessary.

---

## 6. Conclusion

In this notebook, we explored two methods for gender detection:

1. **Pitch Analysis**: By extracting the fundamental frequency and comparing it against a threshold, we achieved a basic form of gender classification.
2. **Machine Learning Classifier**: Using MFCCs and a logistic regression model, we trained a classifier that can generalize to new data.

**Key Takeaways**:

- **Acoustic Features**: Fundamental frequency and MFCCs are powerful features for speech analysis.
- **Statistical Methods**: Thresholding and logistic regression provide different levels of complexity and accuracy.
- **Data Requirements**: Machine learning models require labeled datasets for training.

**Future Work**:

- Collect a larger, more diverse dataset to improve model robustness.
- Explore more advanced models like Support Vector Machines (SVMs) or Neural Networks.
- Incorporate additional features such as formants, energy, and temporal dynamics.

---

## 7. References

1. **Books and Papers**:
   - Rabiner, L., & Juang, B. H. (1993). *Fundamentals of Speech Recognition*. Prentice Hall.
   - Titze, I. R. (1994). *Principles of Voice Production*. Prentice Hall.

2. **Libraries**:
   - **Librosa**: <https://librosa.org/>
   - **Scikit-learn**: <https://scikit-learn.org/>

3. **Datasets**:
   - **Mozilla Common Voice**: <https://commonvoice.mozilla.org/>
   - **VoxCeleb**: <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/>

---
