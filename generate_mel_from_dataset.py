import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectrogram(audio_path, output_image_path, n_mels=128, fmax=8000):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, fmax=fmax)
        plt.axis('off')
        plt.tight_layout(pad=0)

        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {output_image_path}")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def process_audio_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav") or filename.endswith(".mp3"):  # Add other audio formats if needed
            audio_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
            save_mel_spectrogram(audio_path, output_image_path)


# Specify the input and output directories
input_directory = "PMEmo/PMEmo2019/chorus"
output_directory = "dataset/mel_spectrograms"

process_audio_files(input_directory, output_directory)
