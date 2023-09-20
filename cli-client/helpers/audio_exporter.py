import wave


class AudioExporter:
    SUPPORTED_FORMATS = {
        "wav": 0,  # AUDIO_FORMAT_WAV_LPCM_S16LE
        "raw": 1   # AUDIO_FORMAT_RAW_LPCM_S16LE
    }

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def save_audio(self, audio_format: str, audio: bytes, output_filename: str):
        if audio_format == "raw":
            self.__save_audio_raw(audio, output_filename)
        elif audio_format == "wav":
            self.__save_audio_wav(audio, output_filename, self.sample_rate)
        else:
            raise Exception("Could not save resulting audio using provided format.")

    @staticmethod
    def __save_audio_wav(audio: bytes, filename: str, sample_rate: int):
        with wave.open(filename, 'wb') as f:
            f.setsampwidth(2)
            f.setnchannels(1)
            f.setframerate(sample_rate)
            f.writeframesraw(audio)

    @staticmethod
    def __save_audio_raw(audio: bytes, filename: str):
        with open(filename, 'wb') as f:
            f.write(audio)

