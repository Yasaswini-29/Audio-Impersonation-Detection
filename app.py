import streamlit as st
import numpy as np
import librosa
import joblib
import os
import librosa.display
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the models
model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

def preprocess_audio(audio_path):
    """Loads, removes silence, and normalizes audio length."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    # Remove silence
    non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
    audio_trimmed = np.concatenate([audio_data[start:end] for start, end in non_silent_intervals])

    if audio_trimmed.size == 0:
        return None, None

    # Normalize duration to 2 seconds (arbitrary choice)
    target_length = 2 * sr  # 2 seconds
    if len(audio_trimmed) > target_length:
        audio_trimmed = audio_trimmed[:target_length]  # Truncate
    else:
        audio_trimmed = np.pad(audio_trimmed, (0, max(0, target_length - len(audio_trimmed))))  # Pad

    return audio_trimmed, sr

def extract_features(audio_path):
    """Extracts MFCC, Spectrogram, and Spectral Features."""
    audio_data, sr = preprocess_audio(audio_path)
    
    if audio_data is None:
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio_data)

        features = np.hstack((
            np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1), np.mean(chroma, axis=1),
            np.mean(mel_spec_db, axis=1), np.mean(spectral_contrast, axis=1),
            np.mean(spectral_rolloff, axis=1), np.mean(zero_crossing, axis=1)
        ))

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_audio(file_path):
    """Predicts whether the audio is real or fake."""
    features = extract_features(file_path)
    
    if features is not None:
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        confidence = model.predict_proba(features_pca)[0]

        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        confidence_score = max(confidence) * 100
        
        st.write(f"Prediction: {label} | Confidence: {confidence_score:.2f}%")
        plot_spectrogram(file_path)
    else:
        st.write("Error processing the audio file!")

def plot_spectrogram(audio_path):
    """Plots waveform and spectrogram."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    plt.figure(figsize=(10, 5))

    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title("Waveform")

    # Spectrogram
    plt.subplot(2, 1, 2)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
    plt.title("Spectrogram")
    plt.colorbar()

    st.pyplot(plt)

def main():
    st.title("Audio Impersonation Detection")

    st.markdown("""
    This application detects whether an audio file is genuine or fake based on an SVM model trained with MFCC, 
    Spectrogram, and Spectral Features.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file, format="audio/wav")
        
        # Prediction and result display
        predict_audio(file_path)

        # Clean up after prediction
        os.remove(file_path)

if __name__ == "__main__":
    main()
