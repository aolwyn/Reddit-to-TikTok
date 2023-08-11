
#HuggingFace Microsoft Model library requirements
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment

#Google TTS Model Reqs
from gtts import gTTS
from pydub import AudioSegment

#library requirements for the noise reduction
import noisereduce as nr
import librosa
import soundfile as sf

#TODO - GET REDDIT API KEY!

#-----------------------------------------------------------

def test_cuda():
    try:
        if torch.cuda.is_available():
            print("CUDA is available.")
            print("Number of CUDA devices:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))
        else:
            print("CUDA is not available.")
    except Exception as e:
        print("An error occurred while testing CUDA:", e)

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

def TTSGGL(text, language='en', slow=False, output_file='output.mp3'):
    tts = gTTS(text=text, lang=language)
    tts.save(output_file)
    print(f"Google text-to-speech audio saved as {output_file}")

#-----------------------------------------------------------

if __name__ == "__main__":
    Input_Text_Path = "input_text.txt"
    output_Audio_Path = "output.wav"

    with open(Input_Text_Path, "r") as file:
        input_text = file.read()

    test_cuda()
    TTSMS(input_text, output_Audio_Path)
    reduce_noise_in_audio(output_Audio_Path, "noise_reduced_output.wav")
    print("Testing Google Model...")
    TTSGGL(input_text, output_file='output_audio.mp3')
    