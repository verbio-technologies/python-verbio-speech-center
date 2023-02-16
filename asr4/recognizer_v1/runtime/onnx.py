import abc
import logging
import soxr
import numpy as np

import torch
import torch.nn.functional as F
import simple_ctc

import onnx
import onnxruntime
import onnxruntime.quantization
from enum import Enum

from onnxruntime.capi.onnxruntime_pybind11_state import SessionOptions
from onnxruntime.quantization.quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    type_to_name,
    model_has_infer_metadata,
)

from typing import Any, Dict, List, NamedTuple, Optional, Union
from asr4.recognizer_v1.runtime.base import Runtime

MODEL_QUANTIZATION_PRECISION = "INT8"


class DecodingType(Enum):
    GLOBAL = 1
    LOCAL = 2


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


class OnnxSession(Session):
    def __init__(self, path_or_bytes: Union[str, bytes], **kwargs) -> None:
        super().__init__(path_or_bytes)
        self.logger = logging.getLogger("ASR4")
        self.__checkModelWeightPrecision(path_or_bytes)
        self._session = onnxruntime.InferenceSession(
            path_or_bytes,
            sess_options=self.__getSessionOptions(**kwargs),
            providers=kwargs.pop("providers", ["CPUExecutionProvider"]),
            provider_options=kwargs.get("provider_options"),
            **kwargs,
        )
        self.decoding_type = kwargs.pop("decoding_type", DecodingType["GLOBAL"])

    def __getSessionOptions(self, **kwargs) -> SessionOptions:
        session_options = OnnxSession._createSessionOptions(**kwargs)
        self.logger.info(
            f"intra operation number of threads: {session_options.intra_op_num_threads}"
        )
        self.logger.info(
            f"inter operation number of threads: {session_options.inter_op_num_threads}"
        )
        return session_options

    @staticmethod
    def _createSessionOptions(**kwargs) -> SessionOptions:
        options = SessionOptions()
        options.intra_op_num_threads = kwargs.pop("number_of_workers", 0)
        options.inter_op_num_threads = 0 if options.intra_op_num_threads == 0 else 1
        return options

    def __checkModelWeightPrecision(self, path_or_bytes: Union[str, bytes]) -> None:
        model = onnx.load(path_or_bytes)
        if model_has_infer_metadata(model):
            precision = self.__getQuantizationPrecision(model)
            if not precision:
                self.logger.warning(
                    f"Model Quantization Error: expected '{MODEL_QUANTIZATION_PRECISION}' but retrieved 'FLOAT16' weight precision"
                )
            elif precision != MODEL_QUANTIZATION_PRECISION:
                self.logger.warning(
                    f"Model Quantization Error: expected '{MODEL_QUANTIZATION_PRECISION}' but retrieved '{precision}' weight precision"
                )
            else:
                self.logger.info(f"Model quantized - weight precision: '{precision}'")
        else:
            self.logger.warning("Model not quantized - weight precision: 'FLOAT32'")

    def __getQuantizationPrecision(self, model: onnx.ModelProto) -> Optional[str]:
        for node in model.graph.initializer:
            if node.name.endswith(TENSOR_NAME_QUANT_SUFFIX):
                return type_to_name[node.data_type]
        return None

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
            beam_size=1,
            blank_id=0,
            is_nll=False,
        )

    def run(self, input: bytes, sample_rate_hz: int) -> OnnxRuntimeResult:
        if not input:
            raise ValueError("Input audio cannot be empty!")
        x = self._preprocess(input, sample_rate_hz)
        y = self._runOnnxruntimeSession(x)
        return self._postprocess(y)

    @staticmethod
    def _convertToFixedSizeMatrix(audio: np.ndarray, width: int):
        if sizeOfTheLastFrame := audio.shape[0] % width:
            totalToBePadded = width - sizeOfTheLastFrame
            audio = np.pad(audio, (0, totalToBePadded))
        audio = audio.reshape([int(audio.shape[0] / width), width])
        return audio

    def _preprocess(self, input: bytes, sample_rate_hz: int) -> torch.Tensor:
        x = np.frombuffer(input, dtype=np.int16)
        try:
            y = soxr.resample(x, sample_rate_hz, 16000)
        except:
            raise ValueError(f"Invalid audio sample rate: '{sample_rate_hz}'")
        x = y.astype(np.float32)
        if (
            hasattr(self._session, "_session")
            and self._session._session._inputs_meta[0].shape[1] != "sequence_length"
        ):
            x = self._convertToFixedSizeMatrix(
                x, self._session._session._inputs_meta[0].shape[1]
            )
        x = torch.from_numpy(x.copy())
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            x = F.layer_norm(x, x.shape)
        return x

    def _runOnnxruntimeSession(self, input: torch.Tensor) -> _DecodeResult:
        y = []
        label_sequences = []
        scores = []
        timesteps = []

        if len(input.shape) == 2:
            y = self._session.run(None, {self._inputName: input.numpy()})
            return self._decodeTotal(y)
        else:
            return self._batchingDecode(input, label_sequences, scores, timesteps, y)

    def _batchingDecode(self, input, label_sequences, scores, timesteps, y):
        decoding_type = getattr(self._session, "decoding_type", DecodingType.GLOBAL)
        for i in range(input.shape[1]):
            yi = self._session.run(None, {self._inputName: input[:, i, :].numpy()})
            if decoding_type == DecodingType.GLOBAL:
                y += yi
            else:
                label_sequences, scores, timesteps = self._decodePartial(
                    label_sequences, scores, timesteps, yi
                )
        if decoding_type == DecodingType.GLOBAL:
            return self._decodeTotal(y)
        else:
            return _DecodeResult(
                label_sequences=[[label_sequences]],
                scores=[scores],
                timesteps=timesteps,
            )

    def _decodeTotal(self, y):
        y = np.reshape(
            np.vstack([y[i][0, :, :] for i in range(len(y))]),
            [1, len(y) * y[0].shape[1], y[0].shape[2]],
        )
        normalized_y = F.softmax(torch.from_numpy(y), dim=2)
        return self._decoder.decode(normalized_y)

    def _decodePartial(self, label_sequences, scores, timesteps, yi):
        normalized_y = F.softmax(torch.from_numpy(yi[0]), dim=2)
        decoded_part = self._decoder.decode(normalized_y)
        label_sequences += decoded_part.label_sequences[0][0]
        scores += [decoded_part.scores[0][0]]
        timesteps += [decoded_part.timesteps]
        return label_sequences, scores, timesteps

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
