import argparse
from pathlib import Path
import numpy as np
import math
import wave
import sox
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
        help="Gui with the audio paths that must be segmented.",
    )
    parser.add_argument(
        "-a",
        "--audio",
        required=False,
        type=str,
        help="Gui with the audio paths that must be segmented.",
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
    def __init__(self, chunkLength):
        self.chunkLength = chunkLength

    def segmentAudio(
        self,
        audio: dict,
    ):
        if audio["duration"] > self.chunkLength:
            chunks = self.trimAudio(audio)
        elif audio["duration"] < self.chunkLength:
            tfm = sox.Transformer()
            audioData = tfm.build_array(
                input_array=audio["data"], sample_rate_in=audio["sample_rate"]
            )
            chunks = [self.soxPadAudio(audioData, audio["duration"])]
        else:
            chunks = [audio["data"]]
        return chunks

    def trimAudio(self, audio: dict):
        segments = math.ceil(audio["duration"] / self.chunkLength)
        start = 0.0
        audiosTrimmed = []
        for segment in range(segments):
            if start + self.chunkLength < audio["duration"]:
                audiosTrimmed.append(
                    self.soxTrimAudio(audio, start, start + self.chunkLength)
                )
                start = start + self.chunkLength
            else:
                audiosTrimmed.append(self.soxTrimAudio(audio, start, audio["duration"]))
        return audiosTrimmed

    def soxTrimAudio(
        self,
        audio: dict,
        spanStart: float,
        spanEnd: float,
    ):
        audio_length = spanEnd - spanStart

        tfm = sox.Transformer()
        tfm.trim(spanStart, spanEnd)
        audioTrimmed = tfm.build_array(
            input_array=audio["data"], sample_rate_in=audio["sample_rate"]
        )

        if audio_length < self.chunkLength:
            return self.soxPadAudio(audioTrimmed, audio["sample_rate"], audio_length)
        else:
            return audioTrimmed

    def soxPadAudio(self, audio, sample_rate, audio_length: float):
        silence_lenth = self.chunkLength - audio_length
        tfm = sox.Transformer()
        tfm.pad(0, silence_lenth)
        return tfm.build_array(input_array=audio, sample_rate_in=sample_rate)


def loadAudio(audioPath: str):
    audio = {}
    with wave.open(audioPath) as f:
        n = f.getnframes()
        audio["data"] = np.frombuffer(f.readframes(n), dtype=np.int16)
        rate = f.getframerate()
        audio["duration"] = n / float(rate)
        audio["sample_rate"] = rate
    return audio


def saveAudio(audio, sample_rate: int, outputPath: Path):
    scipy.io.wavfile.write(outputPath, sample_rate, audio)


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
            audio = loadAudio(audioFile)
            audioChunks = AudioMultibatching(args.length).segmentAudio(
                loadAudio(audioFile)
            )
            for i, chunk in enumerate(audioChunks):
                saveAudio(
                    chunk,
                    audio["sample_rate"],
                    args.output.joinpath(f"{Path(audioFile).stem}_{i}.wav"),
                )

    if args.audio:
        audio = loadAudio(args.audio)
        audioChunks = AudioMultibatching(args.length).segmentAudio(loadAudio(audio))
        for i, chunk in enumerate(audioChunks):
            saveAudio(
                chunk,
                audio["sample_rate"],
                args.output.joinpath(f"{Path(args.audio).stem}_{i}.wav"),
            )
