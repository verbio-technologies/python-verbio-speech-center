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

    def testGetTimesteps(self):
        decoder = w2l_decoder.W2lKenLMDecoder(
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.bin")),
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            0.2,
            -1,
            0,
        )
        token_idxs = [7, 8, 9, 10]
        self.assertEqual(decoder._getTimesteps([token_idxs]), [0])

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
