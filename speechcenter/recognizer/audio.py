#!/usr/bin/env python3
import wave
import logging
from math import ceil
from options import Options


class Audio:
    def __init__(self, options: Options):
        with open(options.audio_file, "rb") as wav_file:
            wav_data = wave.open(wav_file)
            self.sample_rate = wav_data.getframerate()
            self.audio = wav_data.readframes(wav_data.getnframes())
            wav_data.close()
        self.length = len(self.audio)
    
    def divide(self, chunk_size: int):
        chunk_count = ceil(self.length / chunk_size)
        logging.debug("Dividing audio of length " + str(self.length) + " into " + str(chunk_count) + " of size " + str(chunk_size) + "...")

        if chunk_count > 1:
            for i in range(chunk_count-1):
                start = i*chunk_size
                end = min((i+1)*chunk_size, self.length)

                logging.debug("Audio chunk #" + str(i) + " sliced as " + str(start) + ":" + str(end))
                yield self.audio[start:end]
        else:
            yield self.audio
    
    def get_chunks(self, chunk_size: int = 20000):
        return [chunk for chunk in self.divide(chunk_size)]
