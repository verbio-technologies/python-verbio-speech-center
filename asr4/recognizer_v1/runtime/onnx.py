import abc
import logging
import soxr
import numpy as np
import numpy.typing as npt

import torch
import torch.nn.functional as F

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
from pyformatter import PyFormatter as Formatter
from asr4.recognizer_v1.runtime.w2l_decoder import _DecodeResult

MODEL_QUANTIZATION_PRECISION = "INT8"

LOCAL_FORMATTING_LOOKAHEAD = 1


class DecodingType(Enum):
    GLOBAL = 1
    LOCAL = 2


class OnnxRuntimeResult(NamedTuple):
    sequence: str
    score: float
    wordFrames: List[tuple[int]]
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
        local_formatting: bool = False,
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
            local_formatting,
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
        local_formatting: bool = False,
    ) -> None:
        self.formatter = formatter
        self.local_formatting = local_formatting
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
        self,
        input: bytes,
        sample_rate_hz: int,
        enable_formatting: bool,
    ) -> OnnxRuntimeResult:
        if not input:
            raise ValueError("Input audio cannot be empty!")
        preprocessed_input = self._preprocess(input, sample_rate_hz)
        return self._runOnnxruntimeSession(
            preprocessed_input,
            enable_formatting,
        )

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

    def _runOnnxruntimeSession(
        self, input: torch.Tensor, enable_formatting: bool
    ) -> OnnxRuntimeResult:
        self._session.logger.debug(f" - softmax")
        if len(input.shape) == 2:
            return self._decodeTotal(
                self._session.run(None, {self._inputName: input.numpy()}),
                enable_formatting,
            )
        else:
            return self._batchDecode(
                input,
                enable_formatting,
            )

    def _batchDecode(self, input, enable_formatting: bool) -> OnnxRuntimeResult:
        total_probs = []
        label_sequences = []
        scores = []
        wordFrames = []
        wordTimestamps = []
        chunks_count = 0

        for i in range(input.shape[1]):
            frame_probs = self._session.run(
                None, {self._inputName: input[:, i, :].numpy()}
            )
            if self.decoding_type == DecodingType.GLOBAL:
                total_probs += frame_probs

            elif self.decoding_type == DecodingType.LOCAL and not self.local_formatting:
                (
                    label_sequences,
                    scores,
                    wordFrames,
                    wordTimestamps,
                ) = self._decodePartialAccumulated(
                    label_sequences, scores, wordFrames, wordTimestamps, frame_probs
                )
            elif self.decoding_type == DecodingType.LOCAL and self.local_formatting:
                if i > 0:
                    accumulated_probs += frame_probs
                    y = np.concatenate(accumulated_probs, axis=1)
                else:
                    accumulated_probs = frame_probs
                    y = frame_probs[0]
                (
                    partial_decoding,
                    bufferIndex,
                    chunks_count,
                ) = self._performLocalDecodingWithLocalFormatting(y, chunks_count)
                self._session.logger.info(partial_decoding.sequence)

                if len(bufferIndex) > 0:
                    accumulated_probs = np.concatenate(accumulated_probs, axis=1)
                    accumulated_probs = [
                        np.array(
                            [accumulated_probs[0][bufferIndex[0] : bufferIndex[1]]]
                        )
                    ]

        if self.decoding_type == DecodingType.GLOBAL:
            return self._decodeTotal(total_probs, enable_formatting)

        elif self.decoding_type == DecodingType.LOCAL and self.local_formatting:
            self._runAccumulatedLastChunk(accumulated_probs, chunks_count)
            return partial_decoding  # This is returning only the last partial result. We have to implement how to return each partial.
        else:
            return self._performLocalDecodingWithGlobalFormatting(
                label_sequences, scores, wordFrames, wordTimestamps, enable_formatting
            )

    def _decodeTotal(self, y, enable_formatting) -> OnnxRuntimeResult:
        y = np.concatenate(y, axis=1)
        normalized_y = (
            F.softmax(torch.from_numpy(y), dim=2)
            if self.lmAlgorithm == "viterbi"
            else torch.from_numpy(y)
        )
        self._session.logger.debug(" - decoding global")
        decoding_output = self._decoder.decode(normalized_y)
        postprocessed_output = self._postprocess(decoding_output)
        if enable_formatting:
            return self._performFormatting(postprocessed_output)
        else:
            return postprocessed_output

    def _decodePartialAccumulated(
        self, label_sequences, scores, wordFrames, wordTimestamps, yi
    ):
        normalized_y = F.softmax(torch.from_numpy(yi[0]), dim=2)
        self._session.logger.debug(" - decoding partial with global formatting")
        decoded_part = self._decoder.decode(normalized_y)
        if len(label_sequences) > 0 and self.lmAlgorithm == "kenlm":
            label_sequences += " "
        label_sequences += decoded_part.label_sequences[0][0]
        scores += [decoded_part.scores[0][0]]
        wordFrames += decoded_part.wordsFrames[0][0]
        wordTimestamps += decoded_part.timesteps[0][0]
        return label_sequences, scores, wordFrames, wordTimestamps

    def _decodePartial(self, yi):
        normalized_y = F.softmax(torch.from_numpy(yi), dim=2)
        self._session.logger.debug(" - decoding partial with local formatting")
        decoded_part = self._decoder.decode(normalized_y)
        return _DecodeResult(
            label_sequences=[[decoded_part.label_sequences[0][0]]],
            scores=[[decoded_part.scores[0][0]]],
            wordsFrames=decoded_part.wordsFrames[0][0],
            timesteps=decoded_part.timesteps[0][0],
        )

    def _performLocalDecodingWithGlobalFormatting(
        self, label_sequences, scores, wordFrames, wordTimestamps, enable_formatting
    ):
        decoding_output = _DecodeResult(
            label_sequences=[[label_sequences]],
            scores=[scores],
            wordsFrames=wordFrames,
            timesteps=wordTimestamps,
        )
        postprocessed_output = self._postprocess(decoding_output)
        if enable_formatting:
            return self._performFormatting(postprocessed_output)
        else:
            return postprocessed_output

    def _performLocalDecodingWithLocalFormatting(self, yi, chunks_count):
        saveInBuffer = []
        decoder_result = self._decodePartial(yi)
        postprocessed_output = self._postprocess(decoder_result)
        sequence = ""
        score = 0.0
        wordFrames = ([],)
        wordTimestamps = []
        chunks_count = 0
        if not postprocessed_output.sequence:
            chunks_count += 1
        else:
            formatted_output = self._performFormatting(postprocessed_output)
            self._session.logger.info(formatted_output.sequence)
            formatted_output_until_eos, eos_pos = self._findEOS(formatted_output)
            if eos_pos != -1:
                saveInBuffer = [
                    formatted_output_until_eos.wordFrames[-1][-1] + 1,
                    formatted_output.wordFrames[-1][-1],
                ]
                chunks_count = 0
                sequence = formatted_output_until_eos.sequence
                score = formatted_output_until_eos.score
                wordFrames = formatted_output_until_eos.wordFrames
                wordTimestamps = formatted_output_until_eos.wordTimestamps
            elif chunks_count < LOCAL_FORMATTING_LOOKAHEAD + 1:
                chunks_count += 1
            else:
                chunks_count = 0
                saveInBuffer = []
                sequence = formatted_output.sequence
                score = formatted_output.score
                wordFrames = formatted_output.wordFrames
                wordTimestamps = formatted_output.wordTimestamps
        return (
            OnnxRuntimeResult(
                sequence=sequence,
                score=score,
                wordFrames=wordFrames,
                wordTimestamps=wordTimestamps,
            ),
            saveInBuffer,
            chunks_count,
        )

    def _findEOS(self, formatted_result):
        formatted_tokens = formatted_result.sequence.split(" ")
        selectedTokens = []
        selectedWordFrames = []
        selectedTimestamps = []
        EOS = [".", "?", "!"]
        eos_found = False
        eos_pos = -1

        for pos, token in enumerate(formatted_tokens):
            if token[-1] in EOS and pos != len(formatted_tokens) - 1:
                eos_found = True
                eos_token = pos
                eos_pos = formatted_result.wordFrames[pos][1]

        if eos_found:
            for pos in range(eos_token + 1):
                selectedTokens.append(formatted_tokens[pos])
                selectedWordFrames.append(formatted_result.wordFrames[pos])
                selectedTimestamps.append(formatted_result.wordTimestamps[pos])
        return (
            OnnxRuntimeResult(
                sequence=" ".join(selectedTokens),
                score=formatted_result.score,
                wordFrames=selectedWordFrames,
                wordTimestamps=selectedTimestamps,
            ),
            eos_pos,
        )

    def _runAccumulatedLastChunk(
        self,
        accumulated_probs,
        chunks_count,
    ):
        while len(accumulated_probs) > 0:
            y = np.concatenate(accumulated_probs, axis=1)
            (
                partial_decoding,
                bufferIndex,
                chunks_count,
            ) = self._performLocalDecodingWithLocalFormatting(y, chunks_count)
            self._session.logger.info(partial_decoding.sequence)
            if len(bufferIndex) > 0:
                accumulated_probs = np.concatenate(accumulated_probs, axis=1)
                accumulated_probs = [
                    np.array([accumulated_probs[0][bufferIndex[0] : bufferIndex[1]]])
                ]
            else:
                accumulated_probs = []

    def _postprocess(
        self,
        output: _DecodeResult,
    ) -> OnnxRuntimeResult:
        self._session.logger.debug(" - postprocess")
        sequence = (
            "".join(output.label_sequences[0][0])
            .replace("|", " ")
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
            .strip()
        )
        words = list(filter(lambda x: len(x) > 0, sequence.split(" ")))
        if self.lmAlgorithm == "viterbi":
            score = 1 / np.exp(output.scores[0][0]) if output.scores[0][0] else 0.0
            wordFrames = [(0, 0)] * len(sequence.split(" "))
            timesteps = [(0, 0)] * len(sequence.split(" "))
        else:
            score = output.scores[0][0]
            if self.decoding_type == DecodingType.LOCAL:
                wordFrames = output.wordsFrames
                timesteps = output.timesteps
            else:
                wordFrames = output.wordsFrames[0][0]
                timesteps = output.timesteps[0][0]
        return OnnxRuntimeResult(
            sequence=" ".join(words),
            score=score,
            wordFrames=wordFrames,
            wordTimestamps=timesteps,
        )

    def _performFormatting(self, result: OnnxRuntimeResult) -> OnnxRuntimeResult:
        return OnnxRuntimeResult(
            sequence=self.formatWords(result.sequence),
            score=result.score,
            wordFrames=result.wordFrames,
            wordTimestamps=result.wordTimestamps,  # TODO Implement the wordtimestamps construction from the formatting operations and the original word timestamps
        )

    def formatWords(self, transcription: str) -> str:
        self._session.logger.debug(" - formatting")
        words = list(filter(lambda x: len(x) > 0, transcription.split(" ")))
        if self.formatter and words:
            self._session.logger.debug(f"Pre-formatter text: {words}")
            try:
                return " ".join(self.formatter.classify(words))
            except Exception as e:
                self._session.logger.error(
                    f"Error formatting sentence '{transcription}'"
                )
                self._session.logger.error(e)
        return " ".join(words)


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
