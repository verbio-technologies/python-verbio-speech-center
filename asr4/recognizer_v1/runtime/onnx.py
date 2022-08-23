import torch
import numpy as np
import torch.nn.functional as F
import onnxruntime
import simple_ctc

import abc
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from base import Runtime


class OnnxRuntimeResult(NamedTuple):
    sequence: str
    score: float
    timesteps: List[int]


class _DecodeResult(NamedTuple):
    label_sequences: List[List[List[str]]]
    scores: List[List[float]]
    timesteps: List[List[List[int]]]


class Session(abc.ABC):
    def __init__(
        _path_or_bytes: Union[str, bytes],
        *,
        _sess_options: Optional[onnxruntime.SessionOptions] = None,
        _providers: Optional[Union[List[str], List[Tuple[str, Dict[str, Any]]]]] = None,
        _provider_options: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        raise NotImplementedError()
    
    def run(self,
        _output_names : Optional[List[str]],
        _input_feed : Dict[str, Any],
        *,
        _run_options : Optional[onnxruntime.RunOptions] = None
    ) -> List[np.ndarray]:
        raise NotImplementedError()

    def get_inputs_names(self) -> List[str]:
        raise NotImplementedError()


class OnnxSession(abc.ABC):
    def __init__(
        self,
        path_or_bytes: Union[str, bytes],
        *,
        sess_options: Optional[onnxruntime.SessionOptions] = None,
        providers: Optional[Union[List[str], List[Tuple[str, Dict[str, Any]]]]] = None,
        provider_options: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        self._session = onnxruntime.InferenceSession(
            path_or_bytes, 
            sess_options=sess_options, 
            providers=providers, 
            provider_options=provider_options, 
            **kwargs
        )
    
    def run(self,
        output_names : Optional[List[str]],
        input_feed : Dict[str, Any],
        *,
        run_options : Optional[onnxruntime.RunOptions] = None
    ) -> List[np.ndarray]:
        return self._session.run(output_names, input_feed, run_options)

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
        vocabulary: List[str] = DEFAULT_VOCABULARY
    ) -> None:
        if not session.get_inputs_names():
            raise ValueError(
                "Recognition Model inputs list cannot be empty!"
            )
        self._session = session
        self._inputName = self._session.get_inputs_names()[0]
        self._decoder = simple_ctc.BeamSearchDecoder(
            vocabulary,
            cutoff_top_n=40,
            cutoff_prob=0.8,
            beam_size=100,
            num_processes=1,
            blank_id=0,
            is_nll=True,
        )

    def run(self, input: bytes) -> OnnxRuntimeResult:
        x = self._preprocess(input)
        y = self._runOnnxruntimeSession(x)
        return self._postprocess(y)

    def _preprocess(self, input: bytes) -> torch.Tensor:
        x = np.frombuffer(input, dtype=np.int16)
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy())
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            x = F.layer_norm(x, x.shape)
        return x

    def _runOnnxruntimeSession(self, input: torch.Tensor) -> _DecodeResult:
        y = self._session.run(None, {self._inputName: input.numpy()})
        print(y[0].shape)
        return self._decoder.decode(torch.from_numpy(y[0]))

    def _postprocess(self, output: _DecodeResult) -> OnnxRuntimeResult:
        sequence = "".join(output.label_sequences[0][0]).replace("|", " ").strip()
        timesteps = output.timesteps[0][0] if len(output.timesteps) else []
        return OnnxRuntimeResult(
            sequence=sequence,
            score=output.scores[0][0],
            timesteps=timesteps,
        )


if __name__ == "__main__":
    import wave
    import time
    with wave.open(
        "/home/rmarrugat/Escritorio/verbio/microservices/csr/20111024135504101-12125201610-2980_E_N-read.wav"
    ) as f:
        n = f.getnframes()
        audio_bytes = f.readframes(n)
    start = time.time()
    runtime = OnnxRuntime(
        OnnxSession(
            "/home/rmarrugat/Escritorio/verbio/microservices/csr/asr4/asr4-en-us.onnx"
        )
    )
    print("Init:", time.time() - start)
    start = time.time()
    result = runtime.run(audio_bytes)
    print("Run:", time.time() - start)
    print(result)
