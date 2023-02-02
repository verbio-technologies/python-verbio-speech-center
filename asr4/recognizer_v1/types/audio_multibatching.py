import argparse
from pathlib import Path
import numpy as np
import math
import wave
import sox
import soundfile as sf
import scipy


def _parseArgs():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "-g",
        "--gui",
        required=False,
        type=str,
        help="Gui with the audio paths that must be chunked.",
    )
    parser.add_argument(
        "-a",
        "--audio",
        required=False,
        type=str,
        help="Gui with the audio paths that must be chunked.",
    )
    parser.add_argument(
        "-l",
        "--length",
        required=True,
        type=float,
        help="Length of chunked audios.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output directory where chunked audios will be saved.",
    )
    return parser.parse_args()

class AudioMultibatching:

    def __init__(self, batchLength):
        self.batchLength = batchLength

    def segmentAudio(
        self,
        audio: dict,
    ):
        if audio["duration"] > self.batchLength:
            chunks = self.trimAudio(audio)
        elif audio["duration"] < self.batchLength:
            tfm = sox.Transformer()
            audioData = tfm.build_array(input_array=audio["data"], sample_rate_in=audio["sample_rate"])
            chunks = [ self.soxPadAudio(audioData, audio["duration"]) ]
        else:
            chunks = [ audio["data"] ]
        return chunks

    def trimAudio(self, audio: dict):
        segments = math.ceil(audio["duration"] / self.batchLength)
        start = 0.0
        audiosTrimmed = []
        for segment in range(segments):
            if start + self.batchLength < audio["duration"]:
                audiosTrimmed.append(self.soxTrimAudio(audio, start, start + self.batchLength))
                start = start + self.batchLength
            else:
                audiosTrimmed.append(self.soxTrimAudio(audio, start, audio["duration"]))
        return audiosTrimmed
    
    def soxTrimAudio(self,
        audio: dict, spanStart: float, spanEnd: float,
    ):
        audio_length = spanEnd - spanStart

        tfm = sox.Transformer()
        tfm.trim(spanStart, spanEnd)
        audioTrimmed = tfm.build_array(input_array=audio["data"], sample_rate_in=audio["sample_rate"])

        if audio_length < self.batchLength:
            return self.soxPadAudio(audioTrimmed, audio_length)
        else:
            return audioTrimmed

    def soxPadAudio(self, audio, audio_length: float):
        silence_lenth = self.batchLength - audio_length
        tfm = sox.Transformer()
        tfm.pad(0, silence_lenth)
        return tfm.build_array(input_array=audio, sample_rate_in=8000)

def loadAudio(audioPath: str):
    audio = {}
    with wave.open(audioPath) as f:
        n = f.getnframes()
        audio["data"] = np.frombuffer(f.readframes(n), dtype=np.int16)
        rate = f.getframerate()
        audio["duration"] = n / float(rate)
        audio["sample_rate"] = rate
    return audio

def saveAudio(audio, outputPath: Path):
    scipy.io.wavfile.write(outputPath, 8000, audio)



if __name__ == "__main__":
    args = _parseArgs()

    outputDirectory = args.output
    try:
        outputDirectory.mkdir()
    except FileExistsError:
        print(f"Warning: Folder {outputDirectory} already exists!")

    if args.gui:
        audios = open(args.gui, "r").read().split("\n")
        for audioFile in filter(lambda item: item, audios):
            audioChunks = AudioMultibatching(args.length).segmentAudio(loadAudio(audioFile))
            for i, audio in enumerate(audioChunks):
                saveAudio(audio, args.output.joinpath(f"{Path(audioFile).stem}_{i}.wav"))

    if args.audio:
        audioChunks = AudioMultibatching(args.length).segmentAudio(loadAudio(args.audio))
        for i, audio in enumerate(audioChunks):
            saveAudio(audio, args.output.joinpath(f"{Path(args.audio).stem}_{i}.wav"))


