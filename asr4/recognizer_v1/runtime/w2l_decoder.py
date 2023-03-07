import math
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from flashlight.lib.sequence.criterion import CpuViterbiPath
from flashlight.lib.text.dictionary import create_word_dict, load_words
from flashlight.lib.text.decoder import (
    CriterionType,
    LexiconDecoderOptions,
    KenLM,
    SmearingMode,
    Trie,
    LexiconDecoder,
    DecodeResult,
)

_CUDA_ERROR = ""
LEXICON = Dict[str, List[List[str]]]

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
    ) -> None:
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
        self.nbest = 1
        lexicon = load_words(lexicon)
        self.word_dict = create_word_dict(lexicon)
        lm = KenLM(kenlm_model, self.word_dict)
        trie = self._initializeTrie(vocabulary, lexicon, lm)

        decoder_opts = LexiconDecoderOptions(
            beam_size=5,
            beam_size_token=len(vocabulary),
            beam_threshold=25.0,
            lm_weight=0.2,
            word_score=-1,
            unk_score=-math.inf,
            sil_score=0.0,
            log_add=False,
            criterion_type=CriterionType.CTC,
        )

        self.decoder = LexiconDecoder(
            options=decoder_opts,
            trie=trie,
            lm=lm,
            sil_token_idx=vocabulary.index("|"),
            blank_token_idx=self._blank,
            unk_token_idx=self.word_dict.get_index("<unk>"),
            transitions=[],
            is_token_lm=False,
        )

    def _initializeTrie(
        self,
        vocabulary: List[str],
        lexicon: LEXICON,
        languageModel: KenLM,
    ) -> Trie:
        trie = Trie(len(vocabulary), vocabulary.index("|"))
        startState = languageModel.start(False)
        unkWord = vocabulary.index("<unk>")
        for word, spellings in lexicon.items():
            wordIdx = self.word_dict.get_index(word)
            _, score = languageModel.score(startState, wordIdx)
            for spelling in spellings:
                spellingIdxs = [vocabulary.index(token.lower()) for token in spelling]
                assert (
                    unkWord not in spellingIdxs
                ), f"Some tokens in spelling '{spelling}' were unknown: {spellingIdxs}"
                trie.insert(spellingIdxs, wordIdx, score)
        trie.smear(SmearingMode.MAX)
        return trie

    def decode(self, emissions: npt.NDArray[np.float32]):
        B = emissions.shape[0]
        allWords, allScores, allTimesteps = [], [], []
        for b in range(B):
            hypothesis = self._decodeLexicon(emissions, b)
            words, scores, timesteps = self._postProcessHypothesis(hypothesis)
            allWords.append(words)
            allScores.append(scores)
            allTimesteps.append(timesteps)
        return _DecodeResult(
            label_sequences=allWords, scores=allScores, timesteps=allTimesteps
        )

    def _decodeLexicon(
        self, emissions: npt.NDArray[np.float32], b: int
    ) -> List[DecodeResult]:
        _, T, N = emissions.shape
        emissions_ptr = emissions.ctypes.data + b * (emissions.strides[0] // 2)
        results = self.decoder.decode(emissions_ptr, T, N)
        return results[: self.nbest]

    def _postProcessHypothesis(
        self, hypotesis: List[DecodeResult]
    ) -> Tuple[List[List[str]], List[float], List[List[int]]]:
        words, scores, timesteps = [], [], []
        for result in hypotesis:
            words.append([self.word_dict.get_entry(x) for x in result.words if x >= 0])
            scores.append(result.score)
            timesteps.append(self._getTimesteps(result.tokens))
        return (words, scores, timesteps)

    def _getTimesteps(self, tokenIdxs: List[int]) -> List[int]:
        timesteps = []
        for i, tokenIdx in enumerate(tokenIdxs):
            if tokenIdx == self._blank:
                continue
            if i == 0 or tokenIdx != tokenIdxs[i - 1]:
                timesteps.append(i)
        return timesteps
