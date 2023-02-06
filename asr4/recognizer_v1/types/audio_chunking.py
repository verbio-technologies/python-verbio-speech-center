import argparse
from pathlib import Path
import numpy as np
import math
import wave
import sox
import scipy


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Audio chunking with a specific chunk length."
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


class AudioChunking:
    def __init__(self, chunkLength):
        self.chunkLength = chunkLength

    def segmentAudio(
        self,
        audio: dict,
    ) -> list:
        if audio["duration"] > self.chunkLength:
            chunks = self.trimAudio(audio)
        elif audio["duration"] < self.chunkLength:
            tfm = sox.Transformer()
            audioData = tfm.build_array(
                input_array=audio["data"], sample_rate_in=audio["sample_rate"]
            )
            chunks = [
                self.soxPadAudio(audioData, audio["sample_rate"], audio["duration"])
            ]
        else:
            chunks = [audio["data"]]
        return chunks

    def trimAudio(self, audio: dict) -> list:
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
    try:
        with wave.open(audioPath) as f:
            n = f.getnframes()
            audio["data"] = np.frombuffer(f.readframes(n), dtype=np.int16)
            rate = f.getframerate()
            audio["duration"] = n / float(rate)
            audio["sample_rate"] = rate
    except:
        raise Exception("Audio is empty or does not have good format.")
    return audio


def saveAudio(audio, sample_rate: int, outputPath: Path):
    scipy.io.wavfile.write(outputPath, sample_rate, audio)


def saveAudioChunks(audioChunks, sampleRate: int, outputPath: Path, audioPath: Path):
    for i, chunk in enumerate(audioChunks):
        saveAudio(
            chunk,
            sampleRate,
            outputPath.joinpath(f"{Path(audioPath).stem}_{i}.wav"),
        )


def processGuiFile(guiPath: str, chunksLength: float, outputPath: str):
    audios = open(guiPath, "r").read().split("\n")
    totalAudioChunks = []
    for audioPath in filter(lambda item: item, audios):
        audio = loadAudio(audioPath)
        audioChunks = AudioChunking(chunksLength).segmentAudio(loadAudio(audioPath))
        saveAudioChunks(audioChunks, audio["sample_rate"], outputPath, audioPath)
        totalAudioChunks.append(audioChunks)
    return totalAudioChunks


def processAudioFile(audioPath: str, chunksLength: float, outputPath: str):
    audio = loadAudio(audioPath)
    audioChunks = AudioChunking(chunksLength).segmentAudio(audio)
    saveAudioChunks(audioChunks, audio["sample_rate"], outputPath, audioPath)
    return audioChunks


def main():
    args = parseArgs()

    outputDirectory = args.output
    try:
        outputDirectory.mkdir()
    except FileExistsError:
        print(f"Warning: Folder {outputDirectory} already exists!")

    if args.gui:
        processGuiFile(args.gui, args.length, args.output)

    if args.audio:
        processAudioFile(args.audio, args.length, args.output)


if __name__ == "__main__":
    main()
