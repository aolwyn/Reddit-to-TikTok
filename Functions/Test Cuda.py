from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment
from gtts import gTTS
import noisereduce as nr
import librosa
import soundfile as sf


def test_cuda():
    try:
        if torch.cuda.is_available():
            print("CUDA is available.")
            print("Number of CUDA devices:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))
            print("#-----------------------------------------------------------")
        else:
            print("CUDA is not available.")
            print("#-----------------------------------------------------------")
    except Exception as e:
        print("An error occurred while testing CUDA:", e)
        print("#-----------------------------------------------------------")