import noisereduce as nr
import librosa
import soundfile as sf


file_path = "output.wav"
audio, sr = librosa.load(file_path, sr=None)
reduced_audio = nr.reduce_noise(y=audio, sr=sr)
output_path = "noise_reduced_audio.wav"
sf.write(output_path, reduced_audio, sr)