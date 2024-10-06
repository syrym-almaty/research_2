# Gender Detection

## Gender Detection: Analyze Pitch or Train a Classifier Using Labeled Data

### Introduction

This notebook focuses on gender detection from voice data using both acoustic and machine learning methods. The objective is to explore how the fundamental frequency (pitch), formants, and Mel-frequency cepstral coefficients (MFCCs) can be used to classify gender, with mathematical and statistical justifications. We will also train a machine learning classifier using MFCCs.

---

### 1. **Loading and Visualizing Audio Data**

We begin by loading a `.wav` file and visualizing its waveform. This will give us a basic understanding of the data before extracting features.

#### Code

```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
    """
    Plot the waveform of the audio signal.

    Parameters:
    - y: np.ndarray, audio time series.
    - sr: int, sampling rate.
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform of Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Load and plot
file_path = r'C:\Users\syrym\Downloads\research_2\audio.wav'
y, sr = load_audio(file_path)
plot_waveform(y, sr)
```

#### Explanation

- **Waveform**: The visual representation of the audio signal, where the x-axis represents time and the y-axis represents amplitude.
- **Key Insight**: Male voices typically have lower frequency components, while female voices have higher frequency components.

---

### 2. **Fundamental Frequency (Pitch) Estimation**

We estimate the fundamental frequency (F0), also known as pitch. The pitch is the key feature for distinguishing male and female voices, where males typically have an F0 range of 85-180 Hz and females 165-255 Hz.

#### Code

```python
def extract_pitch(y, sr, fmin=50, fmax=300):
    """
    Extract the fundamental frequency (F0) of the audio signal using the autocorrelation method.

    Parameters:
    - y: np.ndarray, audio time series.
    - sr: int, sampling rate.
    - fmin: float, minimum frequency to consider (Hz).
    - fmax: float, maximum frequency to consider (Hz).

    Returns:
    - pitches: np.ndarray, pitch (F0) values over time.
    """
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=fmin, fmax=fmax)
    
    # Extract pitch for each frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(pitch if pitch > 0 else np.nan)  # NaN for no pitch detected
    
    return np.array(pitch_values)

# Extract pitch
pitches = extract_pitch(y, sr)
```

#### Explanation

- **Autocorrelation Method**: Pitch is estimated by identifying periodicities in the audio signal.
- **Mathematical Formula**:
  \[
  F_0 = \frac{1}{T_0}
  \]
  Where \(T_0\) is the period of vocal fold vibrations.
- **Graphical Insight**: Male voices tend to have a longer period \(T_0\), resulting in lower pitch values.

---

### 3. **Pitch-Based Gender Classification**

We compute the average pitch and use a heuristic to classify gender based on a threshold of 165 Hz.

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
    avg_pitch = np.nanmean(pitches)
    print(f"Average Pitch: {avg_pitch:.2f} Hz")
    
    if avg_pitch > 165:
        return 'Female'
    else:
        return 'Male'

# Classify gender
predicted_gender = classify_gender(pitches)
print(f"Predicted Gender: {predicted_gender}")
```

# Explanation

- **Threshold Approach**: If the average pitch exceeds 165 Hz, the voice is classified as female; otherwise, it is classified as male.
- **Key Observation**: This method works well for clean recordings but may require more sophisticated features in noisy environments.

---

### 4. **Formant Analysis**

Formants are resonant frequencies of the vocal tract, which differ between males and females due to the anatomy of the vocal tract. We estimate the first two formants (F1 and F2).

#### Code

```python
from scipy.signal import lfilter

def extract_formants(y, sr, frame_length=0.025, hop_length=0.01):
    """
    Extract the first two formant frequencies (F1, F2) using Linear Predictive Coding (LPC).

    Parameters:
    - y: np.ndarray, audio time series.
    - sr: int, sampling rate.
    - frame_length: float, frame length (seconds).
    - hop_length: float, hop length (seconds).

    Returns:
    - formants: list of tuples, containing F1 and F2 for each frame.
    """
    n_samples_per_frame = int(frame_length * sr)
    hop_length_samples = int(hop_length * sr)
    formants = []

    for i in range(0, len(y) - n_samples_per_frame, hop_length_samples):
        frame = y[i:i + n_samples_per_frame]
        # Apply Linear Predictive Coding (LPC)
        a = librosa.lpc(frame, 2 + sr // 1000)
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) >= 0]  # Only keep positive frequencies
        formant_frequencies = np.angle(roots) * (sr / (2 * np.pi))

        if len(formant_frequencies) >= 2:
            formants.append((formant_frequencies[0], formant_frequencies[1]))
    
    return formants

# Extract formants
formants = extract_formants(y, sr)
```

#### Explanation

- **Formants**: F1 corresponds to vowel height, while F2 relates to vowel backness. Male speakers generally have lower formants than female speakers due to anatomical differences in the vocal tract.
- **Linear Predictive Coding (LPC)**: LPC is used to model the vocal tract, and formants are the resonant frequencies of this system.

---

### 5. **Mel-Frequency Cepstral Coefficients (MFCCs)**

MFCCs provide a compact representation of the spectral envelope and are commonly used for speech and speaker recognition.

#### Code

```python
def extract_mfccs(y, sr, n_mfcc=13):
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from the audio signal.

    Parameters:
    - y: np.ndarray, audio time series.
    - sr: int, sampling rate.
    - n_mfcc: int, number of MFCCs to extract.

    Returns:
    - mfccs: np.ndarray, MFCC feature matrix.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Extract MFCCs
mfccs = extract_mfccs(y, sr)
```

#### Explanation

- **MFCCs**: These coefficients represent the power spectrum of the audio in a perceptually meaningful way, mimicking the human auditory system.
- **Linear Algebra Insight**: MFCCs are derived by taking the Discrete Cosine Transform (DCT) of the log Mel-scaled power spectrum.

---

### 6. **Training a Classifier for Gender Detection**

We will now train a logistic regression classifier using MFCC features to classify gender. This method will provide a more robust, data-driven solution.

#### Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_gender_classifier(X, y):
    """
    Train a logistic regression model for gender classification based on MFCCs.

    Parameters:
    - X: np.ndarray, feature matrix (MFCCs).
    - y: np.ndarray, gender labels (0 for male, 1 for female).

    Returns:
    - model: trained logistic regression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
   

