import noisereduce as nr
import librosa
import soundfile as sf

def reduceNoise(file_path, output_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        reduced_audio = nr.reduce_noise(y=audio, sr=sr)
        sf.write(output_path, reduced_audio, sr)
        print("Noise reduction completed successfully.")
    except FileNotFoundError:
        print("Error: The input file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inputPath = "output.wav"
    outputPath = "noise_reduced_audio.wav"
    reduceNoise(inputPath, outputPath)