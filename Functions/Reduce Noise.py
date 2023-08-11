from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment
from gtts import gTTS
import noisereduce as nr
import librosa
import soundfile as sf

def reduce_noise_in_audio(input_file_path, output_file_path):
    try:
        print("Starting noise reduction...")
        audio, sr = librosa.load(input_file_path, sr=None)
        reduced_audio = nr.reduce_noise(y=audio, sr=sr)
        sf.write(output_file_path, reduced_audio, sr)
        print("Noise reduction completed successfully.")
    except FileNotFoundError:
        print("Error: The input file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")