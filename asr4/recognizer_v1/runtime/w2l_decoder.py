import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from flashlight.lib.text.dictionary import create_word_dict, load_words
from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from flashlight.lib.text.decoder import (
    CriterionType,
    LexiconDecoderOptions,
    KenLM,
    SmearingMode,
    Trie,
    LexiconDecoder,
)

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


class W2lKenLMDecoder:
    def __init__(
        self,
        useGpu: bool,
        vocabulary: List[str],
        lexicon: str,
        kenlm_model: str,
        decoder_opts: Dict[str, float],
    ) -> None:
        self.vocabulary = vocabulary
        self.criterion_type = CriterionType.CTC

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
        self.silence = vocabulary.index("|")
        self.unk_word = vocabulary.index("<unk>")

        self.nbest = decoder_opts["nbest"]

        self.lexicon = load_words(lexicon)
        self.word_dict = create_word_dict(self.lexicon)
        self.unk_word = self.word_dict.get_index("<unk>")

        self.lm = KenLM(kenlm_model, self.word_dict)

        self.trie = self._initializeTrie()

        self.decoder_opts = LexiconDecoderOptions(
            beam_size=decoder_opts["beam"],
            beam_size_token=decoder_opts["beam_size_token"],
            beam_threshold=decoder_opts["beam_threshold"],
            lm_weight=decoder_opts["lm_weight"],
            word_score=decoder_opts["word_score"],
            unk_score=decoder_opts["unk_weight"],
            sil_score=decoder_opts["sil_weight"],
            log_add=False,
            criterion_type=self.criterion_type,
        )

        self.decoder = LexiconDecoder(
            self.decoder_opts,
            self.trie,
            self.lm,
            self.silence,
            self._blank,
            self.unk_word,
            [],
            False,
        )

    def _initializeTrie(
        self,
    ) -> Trie:
        trie = Trie(len(self.vocabulary), self.silence)
        start_state = self.lm.start(False)
        for word, spellings in self.lexicon.items():
            word_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idxs = [
                    self.vocabulary.index(token.lower()) for token in spelling
                ]
                assert (
                    self.unk_word not in spelling_idxs
                ), f"Some tokens in spelling '{spelling}' were unknown: {spelling_idxs}"
                trie.insert(spelling_idxs, word_idx, score)
        trie.smear(SmearingMode.MAX)
        return trie

    def get_timesteps(self, token_idxs: List[int]) -> List[int]:
        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self._blank:
                continue
            if i == 0 or token_idx != token_idxs[i - 1]:
                timesteps.append(i)
        return timesteps

    def decode(self, emissions: npt.NDArray[np.float32]):
        B = emissions.shape[0]
        results = []
        for b in range(B):
            hypothesis = self._decodeLexicon(emissions, b)
            results.append(hypothesis)

        words, scores, timesteps = self._postProcessHypothesis(hypothesis)
        return _DecodeResult(
            label_sequences=[[" ".join(words)]], scores=[scores], timesteps=[timesteps]
        )

    def _decodeLexicon(self, emissions: npt.NDArray[np.float32], b: int):
        B, T, N = emissions.shape
        emissions_ptr = emissions.ctypes.data + b * (emissions.strides[0] // 2)
        results = self.decoder.decode(emissions_ptr, T, N)
        return results[: self.nbest]

    def _postProcessHypothesis(
        self, hypotesis: List[int]
    ) -> Tuple[List[str], List[float], List[float]]:
        words = []
        scores = []
        timesteps = []
        for result in hypotesis:
            for x in result.words:
                if x >= 0:
                    words.append(self.word_dict.get_entry(x))
            scores.append(result.score)
            timesteps.append(self.get_timesteps(result.tokens))

        return words, scores, timesteps
