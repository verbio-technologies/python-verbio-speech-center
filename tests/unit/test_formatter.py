import pytest, unittest
import os, json
from pathlib import Path

from asr4.recognizer_v1.formatter import *
from asr4.recognizer_v1.runtime.onnx import OnnxRuntimeResult
from asr4.types.language import Language


class TestFormatter(unittest.TestCase):
    def testFormatter(self):
        w = ["my", "email", "is", "l", "formiga", "at", "v", "r", "bio", "dot", "com"]
        f = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.en-us-1.0.1.fm",
            ),
            Language.EN_US,
        )
        (w, ops) = f.classify(w)
        ops = json.loads(ops.to_json()).get("operations", [])
        self.assertEqual(len(ops), 7)
        self.assertEqual(ops[0], {"Change": [9, "."]})
        self.assertEqual(ops[6], {"Change": [3, "Lformiga@vrbio.com"]})


class TestTimestampManagement(unittest.TestCase):
    def testSingleWord(self):
        text = ["hi"]

        wordTimestamps = [(0.0, 0.15)]
        wordFrames = [[1, 4, 8]]
        changeOK = json.loads('{"operations": [{"Change": [0, "."]}]}')["operations"]
        (times, frames) = TimeFixer(changeOK, wordTimestamps, wordFrames).invoke()
        self.assertEqual(times, wordTimestamps)
        self.assertEqual(frames, wordFrames)

        wordTimestamps = [(0.0, 0.15)]
        wordFrames = [[1, 4, 8]]
        changeKO = json.loads(
            '{"operations": [{"Merge": [{"start": 0, "end": 1}, "hi-fi"]}]}'
        )["operations"]
        (times, frames) = TimeFixer(changeKO, wordTimestamps, wordFrames).invoke()
        self.assertEqual(times, wordTimestamps)
        self.assertEqual(frames, wordFrames)

    def testTwoWords(self):
        text = ["pink", "floyd"]

        wordTimestamps = [(0.0, 0.15), (0.18, 0.28)]
        wordFrames = [[1, 4], [10, 16, 22]]
        changeOK = json.loads(
            '{"operations": [{"Merge": [{"start": 0, "end": 1}, "pink-floyd"]}]}'
        )["operations"]
        (times, frames) = TimeFixer(changeOK, wordTimestamps, wordFrames).invoke()
        self.assertEqual(times, [(0.0, 0.28)])
        self.assertEqual(frames, [[1, 4, 10, 16, 22]])

        wordTimestamps = [(0.0, 0.15), (0.18, 0.28)]
        wordFrames = [[1, 4], [10, 16, 22]]
        changeKO = json.loads(
            '{"operations": [{"Merge": [{"start": 2, "end": 5}, "pink-floyd"]}]}'
        )["operations"]
        (times, frames) = TimeFixer(changeKO, wordTimestamps, wordFrames).invoke()
        self.assertEqual(times, wordTimestamps)
        self.assertEqual(frames, wordFrames)

    def testManyWords(self):
        text = ["say", "thirty", "three", "please"]

        wordTimestamps = [(0.0, 0.15), (0.18, 0.28), (0.30, 0.33), (0.39, 0.43)]
        wordFrames = [[1, 4], [10, 16, 22], [25, 29], [30, 31, 32, 33, 34]]
        change1 = json.loads(
            '{"operations": [{"Merge": [{"start": 1, "end": 2}, "33"]}, {"Change": [1, "33,"]}]}'
        )["operations"]
        (times, frames) = TimeFixer(change1, wordTimestamps, wordFrames).invoke()
        self.assertEqual(times, [(0.0, 0.15), (0.18, 0.33), (0.39, 0.43)])
        self.assertEqual(frames, [[1, 4], [10, 16, 22, 25, 29], [30, 31, 32, 33, 34]])

        wordTimestamps = [(0.0, 0.15), (0.18, 0.28), (0.30, 0.33), (0.39, 0.43)]
        wordFrames = [[1, 4], [10, 16, 22], [25, 29], [30, 31, 32, 33, 34]]
        change2 = json.loads(
            '{"operations": [{"Merge": [{"start": 0, "end": 3}, "****"]}]}'
        )["operations"]
        (times, frames) = TimeFixer(change2, wordTimestamps, wordFrames).invoke()
        self.assertEqual(times, [(0.0, 0.43)])
        self.assertEqual(frames, [[1, 4, 10, 16, 22, 25, 29, 30, 31, 32, 33, 34]])

    def testReal(self):
        text = [
            "what",
            "is",
            "the",
            "cost",
            "of",
            "flight",
            "cee",
            "o",
            "one",
            "six",
            "three",
            "one",
            "from",
            "nashville",
            "to",
            "chicago",
        ]
        wordFrames = [
            [18, 19, 21, 22],
            [27, 29],
            [33, 34, 35],
            [40, 44, 49, 52],
            [58, 59],
            [67, 71, 75, 77, 79, 81],
            [100, 104, 108],
            [121],
            [155, 158, 159],
            [168, 172, 176],
            [192, 193, 196, 198, 202],
            [212, 215, 217],
            [229, 231, 233, 235],
            [242, 245, 250, 251, 255, 257, 259, 261, 262],
            [268, 271],
            [276, 277, 280, 284, 288, 295, 297],
        ]
        wordTimestamps = [
            (0.36, 0.44),
            (0.54, 0.58),
            (0.66, 0.7000000000000001),
            (0.8, 1.04),
            (1.16, 1.18),
            (1.34, 1.62),
            (2.0, 2.16),
            (2.42, 2.42),
            (3.1, 3.18),
            (3.36, 3.52),
            (3.84, 4.04),
            (4.24, 4.34),
            (4.58, 4.7),
            (4.84, 5.24),
            (5.36, 5.42),
            (5.5200000000000005, 5.94),
        ]
        ops = json.loads(
            '{"operations": [{"Change": [9, "."]}, {"Merge": [{"start": 6, "end": 10}, "vrbio.com"]}, {"Change": [5, "@"]}, {"Merge": [{"start": 3, "end": 4}, "lformiga"]}, {"Merge": [{"start": 3, "end": 5}, "lformiga@vrbio.com"]}, {"Change": [0, "My"]}, {"Change": [3, "Lformiga@vrbio.com"]}], "original_len": 11}'
        )
        (times, frames) = TimeFixer(
            ops["operations"], wordTimestamps, wordFrames
        ).invoke()

        self.assertEqual(
            times,
            [
                (0.36, 0.44),
                (0.54, 0.58),
                (0.66, 0.7000000000000001),
                (0.8, 4.04),
                (4.24, 4.34),
                (4.58, 4.7),
                (4.84, 5.24),
                (5.36, 5.42),
                (5.5200000000000005, 5.94),
            ],
        )
        self.assertEqual(
            frames,
            [
                [18, 19, 21, 22],
                [27, 29],
                [33, 34, 35],
                [
                    40,
                    44,
                    49,
                    52,
                    58,
                    59,
                    67,
                    71,
                    75,
                    77,
                    79,
                    81,
                    100,
                    104,
                    108,
                    121,
                    155,
                    158,
                    159,
                    168,
                    172,
                    176,
                    192,
                    193,
                    196,
                    198,
                    202,
                ],
                [212, 215, 217],
                [229, 231, 233, 235],
                [242, 245, 250, 251, 255, 257, 259, 261, 262],
                [268, 271],
                [276, 277, 280, 284, 288, 295, 297],
            ],
        )
