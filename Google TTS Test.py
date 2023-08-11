from gtts import gTTS
from pydub import AudioSegment


def TTSGGL(text, language='en', slow=False, output_file='output.mp3'):
    tts = gTTS(text=text, lang=language)
    tts.save(output_file)
    print(f"Text-to-speech audio saved as {output_file}")


if __name__ == "__main__":
    Input_Text_Path = "input_text.txt"

    with open(Input_Text_Path, "r") as file:
        input_text = file.read()

    TTSGGL(input_text, output_file='output_audio.mp3')

