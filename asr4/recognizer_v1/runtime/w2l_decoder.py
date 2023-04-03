import torch
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from flashlight.lib.text.dictionary import create_word_dict, load_words
from flashlight.lib.text.decoder.kenlm import KenLM
from flashlight.lib.text.decoder import (
    CriterionType,
    LexiconDecoderOptions,
    SmearingMode,
    Trie,
    LexiconDecoder,
    DecodeResult,
)

LEXICON = Dict[str, List[List[str]]]


@dataclass
class _DecodeResult:
    label_sequences: List[List[List[str]]]
    scores: List[List[float]] = field(default_factory=lambda: [[0]])
    timesteps: List[List[List[int]]] = field(default_factory=lambda: [[[]]])


class W2lKenLMDecoder:
    def __init__(
        self,
        vocabulary: List[str],
        lmFile: Optional[str],
        lexicon: Optional[str],
        subwords: bool = False,
    ) -> None:
        assert (
            lmFile and lexicon
        ), f"If KenLM is used, neither the language model nor the lexicon can be empty!"

        self._nbest = 1

        self._subwords = subwords

        self.blank = (
            vocabulary.index("<ctc_blank>")
            if "<ctc_blank>" in vocabulary
            else vocabulary.index("<s>")
        )
        if "<sep>" in vocabulary:
            self.silence = vocabulary.index("<sep>")
        elif "|" in vocabulary:
            self.silence = vocabulary.index("|")
        else:
            self.silence = vocabulary.index("</s>")

        lexicon = load_words(lexicon)
        self._wordDict = create_word_dict(lexicon)
        lm = KenLM(lmFile, self._wordDict)
        trie = self._initializeTrie(vocabulary, lm, lexicon)

        decoderOpts = LexiconDecoderOptions(
            beam_size=15,
            beam_size_token=len(vocabulary),
            beam_threshold=25.0,
            lm_weight=0.2,
            word_score=-1,
            unk_score=-np.inf,
            sil_score=0.0,
            log_add=False,
            criterion_type=CriterionType.CTC,
        )

        self._decoder = LexiconDecoder(
            options=decoderOpts,
            trie=trie,
            lm=lm,
            sil_token_idx=self.silence,
            blank_token_idx=self.blank,
            unk_token_idx=self._wordDict.get_index("<unk>"),
            transitions=[],
            is_token_lm=False,
        )

    def _initializeTrie(
        self,
        vocabulary: List[str],
        languageModel: KenLM,
        lexicon: LEXICON,
    ) -> Trie:
        trie = Trie(len(vocabulary), self.silence)
        startState = languageModel.start(False)
        unkWord = vocabulary.index("<unk>")
        for word, spellings in lexicon.items():
            wordIdx = self._wordDict.get_index(word)
            _, score = languageModel.score(startState, wordIdx)
            for spelling in spellings:
                spellingIdxs = [vocabulary.index(token.lower()) for token in spelling]
                assert (
                    unkWord not in spellingIdxs
                ), f"Some tokens in spelling '{spelling}' were unknown: {spellingIdxs}"
                trie.insert(spellingIdxs, wordIdx, score)
        trie.smear(SmearingMode.MAX)
        return trie

    def decode(self, emissions: torch.Tensor):
        B = emissions.shape[0]
        emissions = emissions.numpy()
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
        emissionsPtr = emissions.ctypes.data + b * (emissions.strides[0] // 2)
        results = self._decoder.decode(emissionsPtr, T, N)
        return results[: self._nbest]

    @staticmethod
    def process_word_piece(words: list):
        return "".join(words).replace("_", " ").split()

    def _postProcessHypothesis(
        self, hypotesis: List[DecodeResult]
    ) -> Tuple[List[List[str]], List[float], List[List[int]]]:
        words, scores, timesteps = [], [], []
        for result in hypotesis:
            if not getattr(self, "_subwords", False):
                words.append(
                    " ".join(
                        [self._wordDict.get_entry(x) for x in result.words if x >= 0]
                    )
                )
            else:
                words.append(
                    " ".join(
                        self.process_word_piece(
                            [
                                self._wordDict.get_entry(x)
                                for x in result.words
                                if x >= 0
                            ]
                        )
                    )
                )
            scores.append(result.score)
            timesteps.append(self._getTimesteps(result.tokens))
        return (words, scores, timesteps)

    def _getTimesteps(self, tokenIdxs: List[int]) -> List[int]:
        timesteps = []
        for i, tokenIdx in enumerate(tokenIdxs):
            if tokenIdx == self.blank:
                continue
            if i == 0 or tokenIdx != tokenIdxs[i - 1]:
                timesteps.append(i)
        return timesteps
