import librosa
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
from params import pred_sample_path, pred_mel_path


def cut_sample(input_path, sample_length=45):
    output_path = pred_sample_path
    try:
        audio, sr = librosa.load(input_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    audio_length = len(audio) / sr
    sample_length_samples = sample_length * sr

    if audio_length >= sample_length:
        start_sample = int((len(audio) - sample_length_samples) // 2)
        sample = audio[start_sample:start_sample + sample_length_samples]
    else:
        # If the audio is less than 45s, repeat it to make it 45s long
        repeats = int(np.ceil(sample_length / audio_length))
        extended_audio = np.tile(audio, repeats)
        sample = extended_audio[:sample_length_samples]

    try:
        sf.write(output_path, sample, sr, format='MP3')
    except Exception as e:
        print(f"Error exporting audio file: {e}")


def save_mel_spectrogram(n_mels=128, fmax=8000):
    output_path = pred_mel_path
    try:
        y, sr = librosa.load(pred_sample_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, fmax=fmax)
        plt.axis('off')
        plt.tight_layout(pad=0)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {output_path}: {e}")


# input: path to audio file
# output: path to mel-spectrogram
def preprocess_audio(path_to_audio):
    if os.path.isfile(path_to_audio):
        cut_sample(path_to_audio)
        save_mel_spectrogram()
    else:
        print("ERROR! Invalid path to an audio file")
        raise Exception
