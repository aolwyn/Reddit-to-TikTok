#Documentation:
#https://huggingface.co/microsoft/speecht5_tts

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset
from pydub import AudioSegment

def process_text_to_speech(input_text, output_path):
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

if __name__ == "__main__":
    input_text_path = "input_text.txt"
    output_audio_path = "output.wav"

    with open(input_text_path, "r") as file:
        input_text = file.read()

    process_text_to_speech(input_text, output_audio_path)