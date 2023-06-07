import unittest
from unittest.mock import MagicMock
import pytest
import logging

import torch
import wave
import google
import numpy as np
from random import randint
from typing import Any, Tuple, Dict, List, Optional, Union

from asr4.recognizer_v1.runtime import Session, OnnxRuntime, OnnxSession, DecodingType
from asr4.recognizer_v1.runtime.onnx import _DecodeResult, OnnxRuntimeResult
from asr4.recognizer_v1.loggerService import LoggerService
from asr4.recognizer import Language
from asr4.recognizer_v1.formatter import FormatterFactory
import os


DEFAULT_SPANISH_MESSAGE: str = (
    "hola estoy levantado y en marcha y he recibido un mensaje tuyo"
)
FORMATTED_SPANISH_MESSAGE: str = (
    "Hola. Estoy levantado y en marcha y he recibido un mensaje tuyo."
)


class MockFormatter:
    def __init__(self, correct_sentence: str):
        self._correct_sentence = correct_sentence.split(" ")

    def classify(self, sentence: List[str]) -> List[str]:
        return self._correct_sentence


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
            runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
            runtime.lmAlgorithm = "viterbi"
            runtime.run(b"", 8000, enable_formatting=False)

    def testEmptyInputKenLM(self):
        with self.assertRaises(ValueError):
            runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
            runtime.lmAlgorithm = "kenlm"
            runtime.run(b"", 8000, enable_formatting=False)

    def testRandomInput(self):
        runtime = OnnxRuntime(MockOnnxSession(""))
        result = runtime.run(b"0000", 8000, enable_formatting=False)
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

    def testPostProcessViterbiGlobal(self):
        results = _DecodeResult(
            label_sequences=[
                [["<s>", "h", "e", "l", "l", "o", "<unk>", "<pad>", "</s>"]]
            ],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            timesteps=[[[]]],
        )
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.lmAlgorithm = "viterbi"
        runtime.decoding_type = DecodingType.GLOBAL
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "hello<unk>")
        self.assertEqual(onnxResult.score, 0.0)
        self.assertEqual(onnxResult.wordTimestamps, [(0, 0)])

    def testPostProcessViterbiLocal(self):
        results = _DecodeResult(
            label_sequences=[
                [["<s>", "h", "e", "l", "l", "o", "<unk>", "<pad>", "</s>"]]
            ],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            timesteps=[[[]]],
        )
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.lmAlgorithm = "viterbi"
        runtime.decoding_type = DecodingType.LOCAL
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "hello<unk>")
        self.assertEqual(onnxResult.score, 0.0)
        self.assertEqual(onnxResult.wordTimestamps, [(0, 0)])

    def testPostProcessKenLMGlobal(self):
        results = _DecodeResult(
            label_sequences=[
                [["<s>", "h", "e", "l", "l", "o", "<unk>", "<pad>", "</s>"]]
            ],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            timesteps=[[[(0.2, 1.4)]]],
        )
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.lmAlgorithm = "kenlm"
        runtime.decoding_type = DecodingType.GLOBAL
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "hello<unk>")
        self.assertEqual(onnxResult.score, 0.0)
        self.assertEqual(onnxResult.wordTimestamps, [(0.2, 1.4)])

    def testPostProcessKenLMLocal(self):
        results = _DecodeResult(
            label_sequences=[
                [["<s>", "h", "e", "l", "l", "o", "<unk>", "<pad>", "</s>"]]
            ],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            timesteps=[(0.2, 1.4)],
        )
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.lmAlgorithm = "kenlm"
        runtime.decoding_type = DecodingType.LOCAL
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "hello<unk>")
        self.assertEqual(onnxResult.score, 0.0)
        self.assertEqual(onnxResult.wordTimestamps, [(0.2, 1.4)])

    def testPostProcessNoFramesGlobal(self):
        results = _DecodeResult(
            label_sequences=[[["<s>", "<s>", "<s>", "|", "<s>", "<s>", "<s>", "<s>"]]],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            timesteps=[[[]]],
        )
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.lmAlgorithm = "kenlm"
        runtime.decoding_type = DecodingType.GLOBAL
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "")
        self.assertEqual(onnxResult.score, 0.0)
        self.assertEqual(onnxResult.wordTimestamps, [])

    def testPostProcessNoFramesLocal(self):
        results = _DecodeResult(
            label_sequences=[[["<s>", "<s>", "<s>", "|", "<s>", "<s>", "<s>", "<s>"]]],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            wordsFrames=[],
            timesteps=[],
        )
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.lmAlgorithm = "kenlm"
        runtime.decoding_type = DecodingType.LOCAL
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "")
        self.assertEqual(onnxResult.score, 0.0)
        self.assertEqual(onnxResult.wordTimestamps, [])

    def testFormatter(self):
        sequence = DEFAULT_SPANISH_MESSAGE.split(" ")
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = MockFormatter(FORMATTED_SPANISH_MESSAGE)
        results = OnnxRuntimeResult(
            sequence=" ".join(sequence),
            score=[[(0.0)] * len(sequence)],
            wordFrames=[],
            wordTimestamps=[],
        )
        onnxResult = runtime._performFormatting(results)
        self.assertEqual(
            onnxResult.sequence,
            FORMATTED_SPANISH_MESSAGE,
        )

    def testRecognizeFormatterESNumbers(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.es-es-1.1.0.fm",
            ),
            Language.ES,
        )
        self.assertEqual(
            runtime.formatWords(
                "mi dni es siete siete uno uno cuatro tres seis ocho zeta"
            ),
            "Mi dni es 77114368-Z",
        )

    def testRecognizeFormatterESEmails(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.es-es-1.1.0.fm",
            ),
            Language.ES,
        )
        self.assertEqual(
            runtime.formatWords("mi email es test arroba verbio punto com"),
            "Mi email es Test@verbio.com",
        )

    def testRecognizeFormatterESPunctuation(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.es-es-1.1.0.fm",
            ),
            Language.ES,
        )
        self.assertEqual(
            runtime.formatWords("en qué puedo ayudarle"),
            "¿En qué puedo ayudarle?",
        )

    def testRecognizeFormatterESCapitalization(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.es-es-1.1.0.fm",
            ),
            Language.ES,
        )
        self.assertEqual(
            runtime.formatWords("mi nombre es maría"),
            "Mi nombre es María...",
        )

    def testRecognizeFormatterEN_USNumbers(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.en-us-1.0.1.fm",
            ),
            Language.EN_US,
        )
        self.assertEqual(
            runtime.formatWords("three million dot fourteen"),
            "3,000,000.14.",
        )

    def testRecognizeFormatterEN_USEmails(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.en-us-1.0.1.fm",
            ),
            Language.EN_US,
        )
        self.assertEqual(
            runtime.formatWords("my email address john at gmail dot com"),
            "My email address John@gmail.com.",
        )

    def testRecognizeFormatterEN_USPunctuation(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.en-us-1.0.1.fm",
            ),
            Language.EN_US,
        )
        self.assertEqual(
            runtime.formatWords("how are you"),
            "How are you?",
        )

    def testRecognizeFormatterEN_USCapitalization(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.en-us-1.0.1.fm",
            ),
            Language.EN_US,
        )
        self.assertEqual(
            runtime.formatWords("my name is john"),
            "My name is John.",
        )

    def testRecognizeFormatterPT_BRNumbers(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.pt-br-1.1.1.fm",
            ),
            Language.PT_BR,
        )
        self.assertEqual(
            runtime.formatWords("três mil duzentos e quarenta e cinco"),
            "3.245.",
        )

    def testRecognizeFormatterPT_BREmails(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.pt-br-1.1.1.fm",
            ),
            Language.PT_BR,
        )
        self.assertEqual(
            runtime.formatWords("meu email é joão at domínio dot com"),
            "Meu email é João@domínio.com",
        )

    def testRecognizeFormatterPT_BRPunctuation(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.pt-br-1.1.1.fm",
            ),
            Language.PT_BR,
        )
        self.assertEqual(
            runtime.formatWords("como vai que eu possa ajudar"),
            "Como vai que eu possa ajudar?",
        )

    def testRecognizeFormatterPT_BRCapitalization(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        runtime.formatter = FormatterFactory.createFormatter(
            os.path.join(
                os.getenv("MODELS_PATH", "models"),
                "formatter/format-model.pt-br-1.1.1.fm",
            ),
            Language.PT_BR,
        )
        self.assertEqual(
            runtime.formatWords("meu nome é joão"),
            "Meu nome é João",
        )

    def testFindOneEOSNotFinal(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        formatted_result = OnnxRuntimeResult(
            sequence="Hola. Qué",
            score=1.0,
            wordFrames=[(2, 5), (7, 9)],
            wordTimestamps=[(0.4, 1.0), (1.4, 1.8)],
        )
        partialResult, bufferIndex = runtime._findEOS(formatted_result)
        self.assertEqual(bufferIndex, 5)
        self.assertEqual(partialResult.sequence, "Hola.")
        self.assertEqual(partialResult.score, 1.0)
        self.assertEqual(partialResult.wordFrames, [(2, 5)])
        self.assertEqual(partialResult.wordTimestamps, [(0.4, 1.0)])

    def testFindOneEOSNotFinalWithFinalEOS(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        formatted_result = OnnxRuntimeResult(
            sequence="Hola. ¿Qué tal estás?",
            score=1.0,
            wordFrames=[(2, 5), (7, 9), (11, 13), (15, 18)],
            wordTimestamps=[(0.4, 1.0), (1.4, 1.8), (2.2, 2.6), (3, 3.6)],
        )
        partialResult, bufferIndex = runtime._findEOS(formatted_result)
        self.assertEqual(bufferIndex, 5)
        self.assertEqual(partialResult.sequence, "Hola.")
        self.assertEqual(partialResult.score, 1.0)
        self.assertEqual(partialResult.wordFrames, [(2, 5)])
        self.assertEqual(partialResult.wordTimestamps, [(0.4, 1.0)])

    def testFindMoreThanOneEOSNotFinalWithFinalEOS(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        formatted_result = OnnxRuntimeResult(
            sequence="Hola. ¿Qué tal estás? Me alegro de",
            score=1.0,
            wordFrames=[
                (2, 5),
                (7, 9),
                (11, 13),
                (15, 18),
                (21, 23),
                (25, 29),
                (31, 33),
            ],
            wordTimestamps=[
                (0.4, 1.0),
                (1.4, 1.8),
                (2.2, 2.6),
                (3, 3.6),
                (4.2, 4.6),
                (5, 5.8),
                (6.2, 6.6),
            ],
        )
        partialResult, bufferIndex = runtime._findEOS(formatted_result)
        self.assertEqual(bufferIndex, 18)
        self.assertEqual(partialResult.sequence, "Hola. ¿Qué tal estás?")
        self.assertEqual(partialResult.score, 1.0)
        self.assertEqual(partialResult.wordFrames, [(2, 5), (7, 9), (11, 13), (15, 18)])
        self.assertEqual(
            partialResult.wordTimestamps, [(0.4, 1.0), (1.4, 1.8), (2.2, 2.6), (3, 3.6)]
        )

    def testNotFindEOS(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        formatted_result = OnnxRuntimeResult(
            sequence="Hola qué",
            score=1.0,
            wordFrames=[(2, 5), (7, 9)],
            wordTimestamps=[(0.4, 1.0), (1.5, 1.9)],
        )
        partialResult, bufferIndex = runtime._findEOS(formatted_result)
        self.assertEqual(bufferIndex, -1)
        self.assertEqual(partialResult.sequence, "")
        self.assertEqual(partialResult.score, 1.0)
        self.assertEqual(partialResult.wordTimestamps, [])

    def testPerformLocalDecodingWithLocalFormattingMoreThanOneEOS(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        sequence = "good afternoon thank you for calling bank of america how can i help you today"
        result = _DecodeResult(
            label_sequences=[[sequence]],
            scores=[[1.0]],
            wordsFrames=[(0, 0)] * len(sequence.split(" ")),
            timesteps=[(0, 0)] * len(sequence.split(" ")),
        )
        runtime._decodePartial = MagicMock(return_value=result)
        runtime.formatter = MockFormatter(
            "Good afternoon. Thank you for calling back of America. How can I help you today?"
        )
        (
            result,
            _saveInBuffer,
            _chunksCount,
        ) = runtime._performLocalDecodingWithLocalFormatting([], 0)
        self.assertEqual(
            result.sequence, "Good afternoon. Thank you for calling back of America."
        )

    def testPerformLocalDecodingWithLocalFormattingOneEOS(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        sequence = "good afternoon thank you for calling bank of america"
        result = _DecodeResult(
            label_sequences=[[sequence]],
            scores=[[1.0]],
            wordsFrames=[(0, 0)] * len(sequence.split(" ")),
            timesteps=[(0, 0)] * len(sequence.split(" ")),
        )
        runtime._decodePartial = MagicMock(return_value=result)
        runtime.formatter = MockFormatter(
            "Good afternoon. Thank you for calling back of America."
        )
        (
            result,
            _saveInBuffer,
            _chunksCount,
        ) = runtime._performLocalDecodingWithLocalFormatting([], 0)
        self.assertEqual(result.sequence, "Good afternoon.")

    def testPerformLocalDecodingWithLocalFormattingOneEOS(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        sequence = "good afternoon thank you for calling bank of america"
        result = _DecodeResult(
            label_sequences=[[sequence]],
            scores=[[1.0]],
            wordsFrames=[(0, 0)] * len(sequence.split(" ")),
            timesteps=[(0, 0)] * len(sequence.split(" ")),
        )
        runtime._decodePartial = MagicMock(return_value=result)
        runtime.formatter = MockFormatter(
            "Good afternoon. Thank you for calling back of America."
        )
        (
            result,
            _saveInBuffer,
            _chunksCount,
        ) = runtime._performLocalDecodingWithLocalFormatting([], 0)
        self.assertEqual(result.sequence, "Good afternoon.")

    def testPerformLocalDecodingWithLocalFormattingEOSAtEnd(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        sequence = "good afternoon thank you for calling bank of america"
        result = _DecodeResult(
            label_sequences=[[sequence]],
            scores=[[1.0]],
            wordsFrames=[(0, 0)] * len(sequence.split(" ")),
            timesteps=[(0, 0)] * len(sequence.split(" ")),
        )
        runtime._decodePartial = MagicMock(return_value=result)
        runtime.formatter = MockFormatter("How can I help you.")
        (
            result,
            _saveInBuffer,
            _chunksCount,
        ) = runtime._performLocalDecodingWithLocalFormatting([], 0)
        self.assertEqual(result.sequence, "")

    def testPerformLocalDecodingWithLocalFormattingNoEOS(self):
        runtime = OnnxRuntime(MockOnnxSession(""), "", "", "")
        sequence = "good afternoon thank you for calling bank of america"
        result = _DecodeResult(
            label_sequences=[[sequence]],
            scores=[[1.0]],
            wordsFrames=[(0, 0)] * len(sequence.split(" ")),
            timesteps=[(0, 0)] * len(sequence.split(" ")),
        )
        runtime._decodePartial = MagicMock(return_value=result)
        runtime.formatter = MockFormatter("can I help you")
        (
            result,
            _saveInBuffer,
            _chunksCount,
        ) = runtime._performLocalDecodingWithLocalFormatting([], 0)
        self.assertEqual(result.sequence, "")
