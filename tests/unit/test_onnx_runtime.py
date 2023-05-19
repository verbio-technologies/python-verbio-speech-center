import unittest
import pytest
import logging

import torch
import wave
import google
import numpy as np
from random import randint
from typing import Any, Tuple, Dict, List, Optional, Union

from asr4.recognizer_v1.runtime import Session, OnnxRuntime, OnnxSession
from asr4.recognizer_v1.runtime.onnx import _DecodeResult
from asr4.recognizer_v1.loggerService import LoggerService
from asr4.recognizer import Language


class MockOnnxSession(Session):
    def __init__(self, _path_or_bytes: Union[str, bytes], **kwargs) -> None:
        super().__init__(_path_or_bytes, **kwargs)
        _session_options = kwargs.pop("sess_options", None)
        _providers = kwargs.pop("providers", None)
        self.logger = logging.getLogger("TEST")

    def run(
        self,
        _output_names: Optional[List[str]],
        input_feed: Dict[str, Any],
        **kwargs,
    ) -> np.ndarray:
        batch = input_feed[self.get_inputs_names()[0]].shape[0]
        sequence = randint(1, 5000)
        x = np.random.dirichlet(np.ones(32), size=(batch, sequence))
        x = x.astype(np.float32)
        return [x]

    def get_inputs_names(self) -> List[str]:
        return ["input"]


class TestOnnxSession(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def rootpath(self, pytestconfig):
        self.rootpath = pytestconfig.rootpath
        self.datapath = pytestconfig.rootpath.joinpath("tests/unit/data")

    @pytest.fixture(autouse=True)
    def caplog(self, caplog):
        self.caplog = caplog

    def testNumberOfThreads(self):
        numberOfWorkers = 4
        options = OnnxSession._createSessionOptions(number_of_workers=numberOfWorkers)
        self.assertEqual(1, options.inter_op_num_threads)
        self.assertEqual(numberOfWorkers, options.intra_op_num_threads)

    def testZeroNumberOfThreads(self):
        options = OnnxSession._createSessionOptions(number_of_workers=0)
        self.assertEqual(0, options.inter_op_num_threads)
        self.assertEqual(0, options.intra_op_num_threads)

    def testEmptyModel(self):
        with self.assertRaises(FileNotFoundError):
            _session = OnnxSession("")

    def testInvalidModel(self):
        with self.assertRaises(google.protobuf.message.DecodeError):
            _session = OnnxSession(self.rootpath.joinpath("README.md"))

    def testNonQuantizedModel(self):
        LoggerService.configureLogger(logging.INFO)
        with self.caplog.at_level(logging.WARNING):
            _session = OnnxSession(str(self.datapath.joinpath("mnist-12.onnx")))
        self.assertTrue(
            "Model not quantized - weight precision: 'FLOAT32'" in self.caplog.text
        )

    def testINT8QuantizedModel(self):
        LoggerService.configureLogger(logging.INFO)
        with self.caplog.at_level(logging.INFO):
            _session = OnnxSession(str(self.datapath.joinpath("mnist-12-int8.onnx")))
        self.assertTrue(
            "Model quantized - weight precision: 'INT8'" in self.caplog.text
        )

    def testFLOAT16QuantizedModel(self):
        LoggerService.configureLogger(logging.INFO)
        with self.caplog.at_level(logging.WARNING):
            _session = OnnxSession(str(self.datapath.joinpath("mnist-12-float16.onnx")))
        self.assertTrue(
            "Model Quantization Error: expected 'INT8' but retrieved 'FLOAT16' weight precision"
            in self.caplog.text
        )

    def testUINT8QuantizedModel(self):
        LoggerService.configureLogger(logging.INFO)
        with self.caplog.at_level(logging.WARNING):
            _session = OnnxSession(str(self.datapath.joinpath("mnist-12-uint8.onnx")))
        self.assertTrue(
            "Model Quantization Error: expected 'INT8' but retrieved 'UINT8' weight precision"
            in self.caplog.text
        )


class TestOnnxRuntime(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def rootpath(self, pytestconfig):
        self.rootpath = pytestconfig.rootpath
        self.datapath = pytestconfig.rootpath.joinpath("tests/unit/data")

    def testEmptyInput(self):
        with self.assertRaises(ValueError):
            runtime = OnnxRuntime(MockOnnxSession(""), "", "", "viterbi")
            runtime.run(b"", 8000)

    def testEmptyInputKenLM(self):
        with self.assertRaises(ValueError):
            runtime = OnnxRuntime(MockOnnxSession(""), "", "", "kenlm")
            runtime.run(b"", 8000)

    def testRandomInput(self):
        runtime = OnnxRuntime(MockOnnxSession(""))
        result = runtime.run(b"0000", 8000)
        print(result)
        vocabulary = set(runtime.DEFAULT_VOCABULARY[5:] + [" ", "<", ">"])  # letters
        self.assertEqual(set(result.sequence) - vocabulary, set())
        self.assertTrue(1.0 >= result.score >= 0.0)

    def testPreProcess(self):
        runtime = OnnxRuntime(MockOnnxSession(""))
        tensor = runtime._preprocess(b"0123", 8000)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertTrue(tensor.shape[0], 1)  # batch size
        self.assertTrue(tensor.shape[1], 2)  # n samples
        for language in Language:
            basePath = self.datapath.joinpath(language.value.lower())
            audio, sample_rate = TestOnnxRuntime.__getAudio(
                str(basePath.with_suffix(".8k.wav"))
            )
            tensor8k = runtime._preprocess(audio, sample_rate)
            audio, sample_rate = TestOnnxRuntime.__getAudio(
                str(basePath.with_suffix(".16k.wav"))
            )
            tensor16k = runtime._preprocess(audio, sample_rate)
            torch.testing.assert_close(tensor8k, tensor16k, atol=3.0, rtol=1.3e-6)

    @staticmethod
    def __getAudio(audioFile: str) -> Tuple[bytes, int]:
        with wave.open(audioFile) as f:
            n = f.getnframes()
            audio = f.readframes(n)
            sample_rate_hz = f.getframerate()
        audio = np.frombuffer(audio, dtype=np.int16)
        return (audio.tobytes(), sample_rate_hz)

    def testPostProcess(self):
        results = _DecodeResult(
            label_sequences=[
                [["<s>", "h", "e", "l", "l", "o", "<unk>", "<pad>", "</s>"]]
            ],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            wordTimestamps=[[[]]],
        )
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "viterbi")
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "hello<unk>")
        self.assertEqual(onnxResult.score, 0.0)
