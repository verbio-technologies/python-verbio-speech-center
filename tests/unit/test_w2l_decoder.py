import unittest
import pytest
import torch

from asr4.recognizer_v1.runtime import w2l_decoder


class TestW2lKenLMDecoder(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def rootpath(self, pytestconfig):
        self.rootpath = pytestconfig.rootpath
        self.datapath = pytestconfig.rootpath.joinpath("tests/unit/data")

    @pytest.fixture(autouse=True)
    def vocabulary(self):
        self.vocabulary = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "|",
            "e",
            "t",
            "o",
            "a",
            "i",
            "n",
            "h",
            "s",
            "r",
            "l",
            "d",
            "u",
            "y",
            "w",
            "m",
            "c",
            "g",
            "f",
            "p",
            "b",
            "k",
            "'",
            "v",
            "j",
            "x",
            "q",
            "z",
        ]

    def testGetWordsFrames(self):
        decoder = w2l_decoder.W2lKenLMDecoder(
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.bin")),
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            0.2,
            -1,
            0,
        )
        token_idxs = [
            4,
            6,
            6,
            0,
            14,
            0,
            5,
            4,
            0,
            11,
            0,
            9,
            4,
        ]
        self.assertEqual(
            decoder._getWordsFrames(token_idxs),
            [
                [1, 4, 6],
                [9, 11],
            ],
        )

    def testGetTimeInterval(self):
        decoder = w2l_decoder.W2lKenLMDecoder(
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.bin")),
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            0.2,
            -1,
            0,
        )
        frames = [5, 6, 11, 13, 17]
        self.assertEqual(decoder._getTimeInterval(frames), (0.1, 0.34))

    def testDecode(self):
        decoder = w2l_decoder.W2lKenLMDecoder(
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.bin")),
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            0.2,
            -1,
            0,
        )
        self.assertEqual(
            len(decoder.decode(torch.tensor([[[0, 0, 0]]])).label_sequences), 1
        )
        self.assertEqual(len(decoder.decode(torch.tensor([[[0, 0, 0]]])).scores), 1)
        self.assertEqual(len(decoder.decode(torch.tensor([[[0, 0, 0]]])).timesteps), 1)
