import abc, json, logging, soxr
from enum import Enum
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F

import onnx
import onnxruntime
import onnxruntime.quantization

from onnxruntime.capi.onnxruntime_pybind11_state import SessionOptions
from onnxruntime.quantization.quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    type_to_name,
    model_has_infer_metadata,
)

from asr4.recognizer_v1.runtime.base import Runtime
from asr4.recognizer_v1.runtime.w2l_decoder import _DecodeResult
from asr4.recognizer_v1.formatter import TimeFixer
from pyformatter import PyFormatter as Formatter

MODEL_QUANTIZATION_PRECISION = "INT8"


class DecodingType(Enum):
    GLOBAL = 1
    LOCAL = 2


class OnnxRuntimeResult(NamedTuple):
    sequence: str
    score: float
    wordTimestamps: List[tuple[float]]


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
            return False
        shapes = self.getInputsShapes()
        if not shapes:
            return False
        for shape in shapes:
            if len(shape) == 1 and isinstance(shape[0], str):
                return False
            if len(shape) > 1 and any([isinstance(d, str) for d in shape[1:]]):
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
        self,
        session: Session,
        vocabulary: List[str] = DEFAULT_VOCABULARY,
        formatter: Formatter = None,
        lmFile: Optional[str] = None,
        lexicon: Optional[str] = None,
        lmAlgorithm: str = "viterbi",
        lm_weight: Optional[float] = 0.2,
        word_score: Optional[float] = -1,
        sil_score: Optional[float] = 0.0,
        overlap: Optional[int] = 0,
        subwords: bool = False,
    ) -> None:
        if not session.get_inputs_names():
            raise ValueError("Recognition Model inputs list cannot be empty!")
        self._session = session
        self._inputName = self._session.get_inputs_names()[0]
        self.overlap = overlap
        self._initializeDecoder(
            vocabulary,
            formatter,
            lmFile,
            lexicon,
            lmAlgorithm,
            lm_weight,
            word_score,
            sil_score,
            subwords,
        )

    def _initializeDecoder(
        self,
        vocabulary: List[str],
        formatter: Formatter,
        lmFile: Optional[str],
        lexicon: Optional[str],
        lmAlgorithm: str,
        lm_weight: Optional[float],
        word_score: Optional[float],
        sil_score: Optional[float],
        subwords: bool = False,
    ) -> None:
        self.formatter = formatter
        self.lmAlgorithm = lmAlgorithm
        self.decoding_type = getattr(
            self._session, "decoding_type", DecodingType.GLOBAL
        )
        if lmAlgorithm == "viterbi":
            self._session.logger.debug(f" Using Viterbi algorithm for decoding")
            import simple_ctc

            self._decoder = simple_ctc.BeamSearchDecoder(
                vocabulary,
                cutoff_top_n=32,
                cutoff_prob=0.8,
                beam_size=8,
                blank_id=0,
                is_nll=False,
            )
        elif lmAlgorithm == "kenlm":
            self._session.logger.debug(f" Using KenLM algorithm for decoding")
            from asr4.recognizer_v1.runtime.w2l_decoder import W2lKenLMDecoder

            self._decoder = W2lKenLMDecoder(
                vocabulary, lmFile, lexicon, lm_weight, word_score, sil_score, subwords
            )
        else:
            raise ValueError(
                f"Language Model algorithm should be either 'viterbi' or 'kenlm' but '{lmAlgorithm}' was found."
            )

    def run(
        self, input: bytes, sample_rate_hz: int, enable_formatting
    ) -> OnnxRuntimeResult:
        if not input:
            raise ValueError("Input audio cannot be empty!")
        preprocessed_input = self._preprocess(input, sample_rate_hz)
        decoding_output = self._runOnnxruntimeSession(preprocessed_input)
        return self._postprocess(decoding_output, enable_formatting)

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
                x, self._session._session._inputs_meta[0].shape[1], self.overlap
            )
        x = torch.from_numpy(x.copy())
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            x = F.layer_norm(x, x.shape)
        return x

    @staticmethod
    def _convertToFixedSizeMatrix(
        audio: npt.NDArray[np.float32], width: int, overlap: int
    ):
        # Note that 800 frames are 50ms
        return MatrixOperations(
            window=width, overlap=overlap
        ).splitIntoOverlappingChunks(audio)

    def _runOnnxruntimeSession(self, input: torch.Tensor) -> _DecodeResult:
        self._session.logger.debug(f" - softmax")
        if len(input.shape) == 2:
            y = self._session.run(None, {self._inputName: input.numpy()})
            return self._decodeTotal(y)
        else:
            return self._batchDecode(input)

    def _batchDecode(self, input):
        total_probs = []
        label_sequences = []
        scores = []
        wordTimestamps = []

        for i in range(input.shape[1]):
            frame_probs = self._session.run(
                None, {self._inputName: input[:, i, :].numpy()}
            )
            if self.decoding_type == DecodingType.GLOBAL:
                total_probs += frame_probs
            else:
                label_sequences, scores, wordTimestamps = self._decodePartial(
                    label_sequences, scores, wordTimestamps, frame_probs
                )
        if self.decoding_type == DecodingType.GLOBAL:
            return self._decodeTotal(total_probs)
        else:
            return _DecodeResult(
                label_sequences=[[label_sequences]],
                scores=[scores],
                timesteps=wordTimestamps,
            )

    def _decodeTotal(self, y):
        y = np.concatenate(y, axis=1)
        normalized_y = (
            F.softmax(torch.from_numpy(y), dim=2)
            if self.lmAlgorithm == "viterbi"
            else torch.from_numpy(y)
        )
        self._session.logger.debug(" - decoding global")
        return self._decoder.decode(normalized_y)

    def _decodePartial(self, label_sequences, scores, wordTimestamps, yi):
        normalized_y = F.softmax(torch.from_numpy(yi[0]), dim=2)
        self._session.logger.debug(" - decoding partial")
        decoded_part = self._decoder.decode(normalized_y)
        if len(label_sequences) > 0 and self.lmAlgorithm == "kenlm":
            label_sequences += " "
        label_sequences += decoded_part.label_sequences[0][0]
        scores += [decoded_part.scores[0][0]]
        wordTimestamps += decoded_part.timesteps[0][0]
        return label_sequences, scores, wordTimestamps

    def _postprocess(
        self,
        output: _DecodeResult,
        enable_formatting: bool,
    ) -> OnnxRuntimeResult:
        self._session.logger.debug(" - postprocess")
        (words, timesteps, score) = self._getTimeSteps(output)
        (words, timesteps, frames) = self._performFormatting(words, timesteps, enable_formatting)
        return OnnxRuntimeResult(
            sequence=" ".join(words), score=score, wordTimestamps=timesteps
        )

    def _getTimeSteps(self, output: _DecodeResult):
        text = self._cleanASRoutput(output.label_sequences[0][0])
        words = list(filter(lambda x: len(x) > 0, text.split(" ")))
        if self.lmAlgorithm == "viterbi":
            score = 1 / np.exp(output.scores[0][0]) if output.scores[0][0] else 0.0
            timesteps = [(0, 0)] * len(words)
        else:
            score = output.scores[0][0]
            if self.decoding_type == DecodingType.LOCAL:
                timesteps = output.timesteps
            else:
                timesteps = output.timesteps[0][0]
        return words, timesteps, score

    def _cleanASRoutput(self, sequence):
        return( "".join(sequence)
                .replace("|", " ")
                .replace("<s>", "")
                .replace("</s>", "")
                .replace("<pad>", "")
                .strip())

    def _performFormatting(self, words: str, timesteps, enable_formatting) -> (List[str], List[Any], List[List[int]]):
        if enable_formatting:
            return self.formatWords(words, timesteps)
        else:
            return (words, timesteps, [])

    def formatWords(self, words: str, timesteps: List[List[float]]=None, frames: List[List[int]]=None) -> (List[str], List[Any], List[List[int]]):
        self._session.logger.debug(" - formatting")
        if self.formatter and words:
            self._session.logger.debug(f"Pre-formatter text: {words}")
            try:
                (words, ops) = self.formatter.classify(words.split(" "))
                ops = json.loads(ops.to_json())
                (timesteps, frames) = TimeFixer(ops["operations"], timesteps, frames).invoke()
            except Exception as e:
                self._session.logger.error(
                    f"Error formatting sentence '{words}'"
                )
                self._session.logger.error(e)
        return (" ".join(words), timesteps, frames)


class MatrixOperations:
    def __init__(self, window=160000, overlap=800) -> None:
        self.window = window
        self.overlap = overlap
        self._setCorrectDimensions()

    def splitIntoOverlappingChunks(self, audio: npt.NDArray[np.float32]):
        overlap_size = self.window - self.overlap
        num_chunks = int(np.ceil((audio.shape[0]) / overlap_size))
        result = np.zeros((num_chunks, self.window), dtype=audio.dtype)

        # Iterate over the chunks
        for i in range(num_chunks):
            # Calculate the start and end indices for the current chunk
            start_idx = i * overlap_size
            end_idx = start_idx + self.window

            # Extract the current chunk from the input array
            chunk = audio[start_idx:end_idx]

            if i and i == (num_chunks - 1) and self.overlap >= chunk.shape[0]:
                # Delete last row as it has no information
                result = result[:-1, :]
            else:
                # Store the chunk in the result array
                result[i, : chunk.shape[0]] = chunk

        return result

    def _setCorrectDimensions(self):
        assert (
            self.overlap <= self.window and self.window
        ), "Cannot split into overlapping chunks if overlap is bigger than window"
