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


class MockW2lKenLMDecoder:
    def __init__(
        self,
        useGpu: bool,
        vocabulary: List[str],
        lexicon: str,
        kenlm_model: str,
        decoder_opts: Dict[str, float],
        **kwargs,
    ) -> None:
        self.kenlm_model = kenlm_model
        self.lexicon = lexicon
        self.vocabulary = vocabulary
        self.gpu = useGpu
        session_options = kwargs.pop("sess_options", None)
        providers = kwargs.pop("providers", None)


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

    @pytest.fixture(autouse=True)
    def decoderOpts(self):
        self.decoder_opts = {
            "beam": 5,
            "beam_size_token": 32,
            "beam_threshold": 25.0,
            "lm_weight": 0.2,
            "word_score": -1,
            "unk_weight": -math.inf,
            "sil_weight": 0.0,
            "nbest": 1,
        }

    def testGetTimesteps(self):
        decoder = w2l_decoder.W2lKenLMDecoder(
            False,
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            str(self.datapath.joinpath("en-us_lm.bin")),
            self.decoder_opts,
        )
        token_idxs = [7, 8, 9, 10]
        self.assertEqual(decoder.get_timesteps([token_idxs]), [0])

    def testDecode(self):
        decoder = w2l_decoder.W2lKenLMDecoder(
            False,
            self.vocabulary,
            str(self.datapath.joinpath("en-us_lm.lexicon.txt")),
            str(self.datapath.joinpath("en-us_lm.bin")),
            self.decoder_opts,
        )
        self.assertEqual(
            len(decoder.decode(np.array([[[0, 0, 0]]])).label_sequences), 1
        )
        self.assertEqual(len(decoder.decode(np.array([[[0, 0, 0]]])).scores), 1)
        self.assertEqual(len(decoder.decode(np.array([[[0, 0, 0]]])).timesteps), 1)
