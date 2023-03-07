import unittest
import pytest
import logging

import torch
import wave
import google
import numpy as np
from random import randint
from typing import Any, Tuple, Dict, List, Optional, Union

from asr4.recognizer_v1.runtime import w2l_decoder
from asr4.recognizer_v1.runtime.onnx import _DecodeResult

import torch
import torch.nn.functional as F


import math


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
            False,
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            str(self.datapath.joinpath("en-us_lm.bin")),
        )
        token_idxs = [7, 8, 9, 10]
        self.assertEqual(decoder.get_timesteps([token_idxs]), [0])

    def testDecode(self):
        decoder = w2l_decoder.W2lKenLMDecoder(
            False,
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            str(self.datapath.joinpath("en-us_lm.bin")),
        )
        self.assertEqual(
            len(decoder.decode(np.array([[[0, 0, 0]]])).label_sequences), 1
        )
        self.assertEqual(len(decoder.decode(np.array([[[0, 0, 0]]])).scores), 1)
        self.assertEqual(len(decoder.decode(np.array([[[0, 0, 0]]])).timesteps), 1)
