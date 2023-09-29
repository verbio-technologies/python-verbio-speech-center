import wave


class AudioImporter:
    def __init__(self, audio_file: str):
        with open(audio_file, "rb") as wav_file:
            wav_data = wave.open(wav_file)
            self.sample_rate = wav_data.getframerate()
            self.audio = wav_data.readframes(wav_data.getnframes())
            wav_data.close()

