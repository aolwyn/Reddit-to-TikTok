#HuggingFace Microsoft Model library requirements
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment

#Google TTS Model Reqs
from gtts import gTTS
#from pydub import AudioSegment <-- duplicated, see above.

#library requirements for the noise reduction
import noisereduce as nr
import librosa
import soundfile as sf

#For API things
from dotenv import load_dotenv
import os

#helpful documentation:
#https://github.com/jiaaro/pydub
#https://huggingface.co/microsoft/speecht5_tts
#https://github.com/MiniGlome/Tiktok-uploader


#TODO - GET REDDIT API KEY!

#-----------------------------------------------------------

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

#-----------------------------------------------------------

def clean_text(input_text):
    try:
        cleaned_text = input_text.replace("\"", "")
        cleaned_text = cleaned_text.replace("AITA", "am I the A Hole")
        cleaned_text = cleaned_text.replace(".", " ")
        
        return cleaned_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#-----------------------------------------------------------

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

#-----------------------------------------------------------

def TTSGGL(text, language='en', slow=False, output_file='TTS_GGL.mp3'):
    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        tts.save(output_file)
        print(f"Google text-to-speech audio saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

#-----------------------------------------------------------

def increase_playback_speed(input_path, output_path, speed_factor=1.1):
    try:
        song = AudioSegment.from_mp3(input_path)
        
        spedup = song.speedup(playback_speed=speed_factor, chunk_size=150, crossfade=25)
        spedup.export(output_path, format="mp3")
        
        print("Playback speed increased and saved successfully.")
    except Exception as e:
        print("An error occurred:", e)

#-----------------------------------------------------------

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

#-----------------------------------------------------------

if __name__ == "__main__":

    #set the location of the input text
    input_text_path = "input_text.txt"
    
    #open said text
    with open(input_text_path, "r") as file:
        input_text = file.read()

    #load .env variables 
    load_dotenv()

    #test tensors + run 
    test_cuda()
    clean_text(input_text)
    TTSGGL(input_text,output_file='TTS_GGL.mp3')
    