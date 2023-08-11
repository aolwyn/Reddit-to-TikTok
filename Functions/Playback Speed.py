from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment
from gtts import gTTS
import noisereduce as nr
import librosa
import soundfile as sf

def increase_playback_speed(input_path, output_path, speed_factor=1.1):
    try:
        song = AudioSegment.from_mp3(input_path)
        
        spedup = song.speedup(playback_speed=speed_factor, chunk_size=150, crossfade=25)
        spedup.export(output_path, format="mp3")
#         print({
#     'duration' : song.duration_seconds,
#     'sample_rate' : song.frame_rate,
#     'channels' : song.channels,
#     'sample_width' : song.sample_width,
#     'frame_count' : song.frame_count(),
#     'frame_rate' : song.frame_rate,
#     'frame_width' : song.frame_width,
# })
        print("Playback speed increased and saved successfully.")
    except Exception as e:
        print("An error occurred:", e)