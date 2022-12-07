import torch
import numpy as np
import resampy
import torch.nn.functional as F
import onnxruntime
import simple_ctc

import abc
from typing import Any, Dict, List, NamedTuple, Optional, Union
from asr4.recognizer_v1.runtime.base import Runtime


class OnnxRuntimeResult(NamedTuple):
    sequence: str
    score: float


class _DecodeResult(NamedTuple):
    label_sequences: List[List[List[str]]]
    scores: List[List[float]]
    timesteps: List[List[List[int]]]


class Session(abc.ABC):
    def __init__(
        self,
        _path_or_bytes: Union[str, bytes],
        **kwargs,
    ) -> None:
        pass

    def run(
        self,
        _output_names: Optional[List[str]],
        _input_feed: Dict[str, Any],
        **kwargs,
    ) -> List[np.ndarray]:
        raise NotImplementedError()

    def get_inputs_names(self) -> List[str]:
        raise NotImplementedError()


class OnnxSession(abc.ABC):
    def __init__(self, path_or_bytes: Union[str, bytes], **kwargs) -> None:
        providers = kwargs.get("providers")
        del kwargs["providers"]
        self._session = onnxruntime.InferenceSession(
            path_or_bytes,
            sess_options=kwargs.get("sess_options"),
            providers=providers,
            provider_options=kwargs.get("provider_options"),
            **kwargs,
        )

    def run(
        self,
        output_names: Optional[List[str]],
        input_feed: Dict[str, Any],
        **kwargs,
    ) -> List[np.ndarray]:
        return self._session.run(output_names, input_feed, kwargs.get("run_options"))

    def get_inputs_names(self) -> List[str]:
        return [input.name for input in self._session.get_inputs()]


class OnnxRuntime(Runtime):
    DEFAULT_VOCABULARY: List[str] = [
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

    def __init__(
        self, session: Session, vocabulary: List[str] = DEFAULT_VOCABULARY
    ) -> None:
        if not session.get_inputs_names():
            raise ValueError("Recognition Model inputs list cannot be empty!")
        self._session = session
        self._inputName = self._session.get_inputs_names()[0]
        self._decoder = simple_ctc.BeamSearchDecoder(
            vocabulary,
            cutoff_top_n=32,
            cutoff_prob=0.8,
            beam_size=100,
            blank_id=0,
            is_nll=False,
        )

    def run(self, input: bytes, sample_rate_hz: int) -> OnnxRuntimeResult:
        if not input:
            raise ValueError("Input audio cannot be empty!")
        x = self._preprocess(input, sample_rate_hz)
        y = self._runOnnxruntimeSession(x)
        return self._postprocess(y)

    def _preprocess(self, input: bytes, sample_rate_hz: int) -> torch.Tensor:
        x = np.frombuffer(input, dtype=np.int16)
        try:
            y = resampy.resample(x, sample_rate_hz, 16000)
        except:
            raise ValueError(f"Invalid audio sample rate: '{sample_rate_hz}'")
        x = y.astype(x.dtype)
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy())
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            x = F.layer_norm(x, x.shape)
        return x

    def _runOnnxruntimeSession(self, input: torch.Tensor) -> _DecodeResult:
        y = self._session.run(None, {self._inputName: input.numpy()})
        normalized_y = F.softmax(torch.from_numpy(y[0]), dim=2)
        return self._decoder.decode(normalized_y)

    @staticmethod
    def _postprocess(output: _DecodeResult) -> OnnxRuntimeResult:
        sequence = (
            "".join(output.label_sequences[0][0])
            .replace("|", " ")
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
            .strip()
        )
        score = 1 / np.exp(output.scores[0][0]) if output.scores[0][0] else 0.0
        return OnnxRuntimeResult(
            sequence=sequence,
            score=score,
        )
