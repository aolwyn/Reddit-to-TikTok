from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment
from gtts import gTTS
import noisereduce as nr
import librosa
import soundfile as sf

def TTSMS(input_text, output_path):
    print("Testing Microsoft TTS Hugging Face Model...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=input_text, return_tensors="pt")

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    print("Writing Text to Speech Wav file...")
    sf.write(output_path, speech.numpy(), samplerate=16500)
    print("Complete!")