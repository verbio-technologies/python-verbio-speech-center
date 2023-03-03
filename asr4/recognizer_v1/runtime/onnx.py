import abc
import logging
import soxr
import numpy as np
import numpy.typing as npt

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

    def getInputsShapes(self) -> List[List[Union[int, str]]]:
        return [input.shape for input in self._session.get_inputs()]

    def isModelShapeStatic(self) -> bool:
        if not hasattr(self, "_session"):
            print("-> has attr session")
            return False
        shapes = self.getInputsShapes()
        if not shapes:
            print("-> not shapes")
            return False
        for shape in shapes:
            if len(shape) == 1 and isinstance(shape[0], str):
                print("-> len(shape)==1")
                return False
            if len(shape) > 1 and any([isinstance(d, str) for d in shape[1:]]):
                print("-> len(shape)>1")
                return False
        return True


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
            beam_size=8,
            blank_id=0,
            is_nll=False,
        )

    def run(self, input: bytes, sample_rate_hz: int) -> OnnxRuntimeResult:
        if not input:
            raise ValueError("Input audio cannot be empty!")
        x = self._preprocess(input, sample_rate_hz)
        y = self._runOnnxruntimeSession(x)
        self._session.logger.debug(" - postprocess")
        return self._postprocess(y)

    def _preprocess(self, input: bytes, sample_rate_hz: int) -> torch.Tensor:
        self._session.logger.debug(f" - preprocess audio of length {len(input)}")
        x = np.frombuffer(input, dtype=np.int16)
        try:
            y = soxr.resample(x, sample_rate_hz, 16000)
        except:
            raise ValueError(f"Invalid audio sample rate: '{sample_rate_hz}'")
        x = y.astype(np.float32)
        if self._session.isModelShapeStatic():
            self._session.logger.debug(" - split chunks")
            x = self._convertToFixedSizeMatrix(
                x, self._session._session._inputs_meta[0].shape[1]
            )
        x = torch.from_numpy(x.copy())
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            x = F.layer_norm(x, x.shape)
        return x

    @staticmethod
    def _convertToFixedSizeMatrix(audio: npt.NDArray[np.float32], width: int):
        #800 frames are 50ms
        return MatrixOperations(window=width, overlap=800).splitIntoOverlappingChunks(audio)

    def _runOnnxruntimeSession(self, input: torch.Tensor) -> _DecodeResult:
        self._session.logger.debug(f" - decoding {len(input.shape)}")
        if len(input.shape) == 2:
            y = self._session.run(None, {self._inputName: input.numpy()})
            return self._decodeTotal(y)
        else:
            return self._batchDecode(input)

    def _batchDecode(self, input):
        decoding_type = getattr(self._session, "decoding_type", DecodingType.GLOBAL)

        total_probs = []
        label_sequences = []
        scores = []
        timesteps = []

        for i in range(input.shape[1]):
            frame_probs = self._session.run(
                None, {self._inputName: input[:, i, :].numpy()}
            )
            if decoding_type == DecodingType.GLOBAL:
                total_probs += frame_probs
            else:
                label_sequences, scores, timesteps = self._decodePartial(
                    label_sequences, scores, timesteps, frame_probs
                )

        if decoding_type == DecodingType.GLOBAL:
            return self._decodeTotal(total_probs)
        else:
            return _DecodeResult(
                label_sequences=[[label_sequences]],
                scores=[scores],
                timesteps=timesteps,
            )

    def _decodeTotal(self, y):
        y = np.concatenate(y, axis=1)
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

class MatrixOperations:
    def __init__(self, window=160000, overlap=800) -> None:
        self.window = window
        self.overlap = overlap

    def splitIntoOverlappingChunks(self, audio: npt.NDArray[np.float32]):
        self._setCorrectDimesions()
        audio = self._makeIntoMatrix(audio)
        return self._addRightPadding(audio)

    def _setCorrectDimesions(self):
        self.window = self.window - 2*self.overlap
        assert self.window>0, "Can not split into overlapping chunks if overlap is bigger than window"

    def _makeIntoMatrix(self, audio: npt.NDArray[np.float32]):
        sizeOfTheLastFrame = (self.overlap + audio.shape[0]) % self.window
        totalToBePadded = self.window - sizeOfTheLastFrame
        if sizeOfTheLastFrame == 0:
            totalToBePadded = 0
        audio = np.pad(audio, (self.overlap, totalToBePadded))
        return audio.reshape([audio.shape[0] // self.window, self.window])

    def _addRightPadding(self, audio: npt.NDArray[np.float32]):
        repetition = audio[1:,0:2*self.overlap]
        repetition = np.vstack(( repetition, np.zeros((1,repetition.shape[1]), dtype=audio.dtype) ))
        return np.hstack((audio, repetition))
