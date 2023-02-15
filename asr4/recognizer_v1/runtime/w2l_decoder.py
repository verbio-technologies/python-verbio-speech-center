import struct
import itertools
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import List, Any

from flashlight.lib.sequence.criterion import CpuViterbiPath

_CUDA_ERROR = ""

try:
    from flashlight.lib.sequence.flashlight_lib_sequence_criterion import (
        CudaViterbiPath,
    )
except Exception as e:
    _CUDA_ERROR = f"Could not load Flashlight Sequence CudaViterbiPath: {e}"


@dataclass
class _DecodeResult:
    label_sequences: List[List[List[str]]]
    scores: List[List[float]] = field(default_factory=lambda: [[0]])
    timesteps: List[List[List[int]]] = field(default_factory=lambda: [[[]]])


class W2lViterbiDecoder:
    def __init__(self, useGpu: bool, vocabulary: List[str]) -> None:
        self._vocabulary = vocabulary

        if useGpu:
            if _CUDA_ERROR:
                raise Exception(_CUDA_ERROR)
            self._flashlight = CudaViterbiPath
        else:
            self._flashlight = CpuViterbiPath

        self._blank = (
            vocabulary.index("<pad>")
            if "<pad>" in vocabulary
            else vocabulary.index("<s>")
        )

    def decode(self, emissions: npt.NDArray[np.float32]) -> _DecodeResult:
        batchSize = emissions.shape[0]
        viterbiPath = self._computeViterbi(emissions)
        r = []
        for idx in range(batchSize):
            hypotesis = self._ctc(viterbiPath[idx].tolist())
            hypotesis = self._postProcessHypothesis(hypotesis)
            r.append(hypotesis)
        return _DecodeResult(label_sequences=[r])

    def _computeViterbi(
        self, emissions: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        B, T, N = emissions.shape
        transitions = np.zeros([N, N], dtype=np.float32)
        viterbiPath = np.empty([B, T], dtype=np.int32)
        workspace = np.array(
            self._flashlight.get_workspace_size(B, T, N), dtype=np.uint8
        )
        self._flashlight.compute(
            B,
            T,
            N,
            W2lViterbiDecoder.getDataPtrAsBytes(emissions),
            W2lViterbiDecoder.getDataPtrAsBytes(transitions),
            W2lViterbiDecoder.getDataPtrAsBytes(viterbiPath),
            W2lViterbiDecoder.getDataPtrAsBytes(workspace),
        )
        return viterbiPath

    @staticmethod
    def getDataPtrAsBytes(tensor: npt.NDArray[Any]) -> bytes:
        return struct.pack("P", tensor.__array_interface__["data"][0])

    def _ctc(self, hypotesis: List[int]) -> List[int]:
        hypotesis = (g[0] for g in itertools.groupby(hypotesis))
        hypotesis = filter(lambda x: x != self._blank, hypotesis)
        return list(hypotesis)

    def _postProcessHypothesis(self, hypotesis: List[int]) -> List[str]:
        return [
            self._vocabulary[letter]
            for letter in hypotesis
            if letter < len(self._vocabulary)
        ]
