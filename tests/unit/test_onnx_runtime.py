import unittest
from asr4.recognizer_v1.runtime import Session, OnnxRuntime
from asr4.recognizer_v1.runtime.onnx import _DecodeResult

import torch
import numpy as np
import onnxruntime
from random import randint
from typing import Any, Dict, List, Optional, Tuple, Union


class MockOnnxSession(Session):
    def __init__(
        self,
        _path_or_bytes: Union[str, bytes],
        *,
        _sess_options: Optional[onnxruntime.SessionOptions] = None,
        _providers: Optional[Union[List[str], List[Tuple[str, Dict[str, Any]]]]] = None,
        _provider_options: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        pass

    def run(
        self,
        _output_names: Optional[List[str]],
        input_feed: Dict[str, Any],
        *,
        _run_options: Optional[onnxruntime.RunOptions] = None
    ) -> np.ndarray:
        batch = input_feed[self.get_inputs_names()[0]].shape[0]
        sequence = randint(1, 5000)
        x = np.random.dirichlet(np.ones(32), size=(batch, sequence))
        x = x.astype(np.float32)
        return [x]

    def get_inputs_names(self) -> List[str]:
        return ["input"]


class TestOnnxRuntime(unittest.TestCase):
    def testEmptyInput(self):
        with self.assertRaises(ValueError):
            runtime = OnnxRuntime(MockOnnxSession(""))
            runtime.run(b"")

    def testRandomInput(self):
        runtime = OnnxRuntime(MockOnnxSession(""))
        result = runtime.run(b"0000")
        vocabulary = set(runtime.DEFAULT_VOCABULARY[5:] + [" ", "<", ">"])  # letters
        self.assertEqual(set(result.sequence) - vocabulary, set())
        self.assertTrue(result.score <= 1.0 and result.score >= 0.0)

    def testPreProcess(self):
        runtime = OnnxRuntime(MockOnnxSession(""))
        tensor = runtime._preprocess(b"0123")
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertTrue(tensor.shape[0], 1)  # batch size
        self.assertTrue(tensor.shape[1], 2)  # n samples

    def testPostProcess(self):
        results = _DecodeResult(
            label_sequences=[
                [["<s>", "h", "e", "l", "l", "o", "<unk>", "<pad>", "</s>"]]
            ],
            scores=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            timesteps=[[[]]],
        )
        runtime = OnnxRuntime(MockOnnxSession(""))
        onnxResult = runtime._postprocess(results)
        self.assertEqual(onnxResult.sequence, "hello<unk>")
        self.assertEqual(onnxResult.score, 0)
