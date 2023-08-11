from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment
from gtts import gTTS
import noisereduce as nr
import librosa
import soundfile as sf

def TTSGGL(text, language='en', slow=False, output_file='TTS_GGL.mp3'):
    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        tts.save(output_file)
        print(f"Google text-to-speech audio saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")