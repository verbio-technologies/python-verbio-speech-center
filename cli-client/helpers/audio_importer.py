import os
import subprocess
import wave

def preprocess_audio_file_to_pcm(audio_file: str):
    tmp_audio_file = "./" + os.path.basename(audio_file) + "_tmp.wav"
    command = "sox " + audio_file + " -e signed-integer " + tmp_audio_file
    subprocess.run(command.split())
    return tmp_audio_file


def remove_pcm_audio_file(audio_file: str):
    os.remove(audio_file)


class AudioImporter:
    def __init__(self, audio_file: str):
        tmp_audio_file = preprocess_audio_file_to_pcm(audio_file)
        with open(tmp_audio_file, "rb") as wav_file:
            wav_data = wave.open(wav_file)
            self.sample_rate = wav_data.getframerate()
            self.audio = wav_data.readframes(wav_data.getnframes())
            wav_data.close()
        remove_pcm_audio_file(tmp_audio_file)
