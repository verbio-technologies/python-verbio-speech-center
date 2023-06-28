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
    wordsFrames: List[List[List[List[int]]]] = field(
        default_factory=lambda: [[[(0, 0)]]]
    )
    scores: List[List[float]] = field(default_factory=lambda: [[0]])
    timesteps: List[List[List[tuple[float]]]] = field(
        default_factory=lambda: [[[(0, 0)]]]
    )


class W2lKenLMDecoder:
    def __init__(
        self,
        vocabulary: List[str],
        lmFile: Optional[str],
        lexicon: Optional[str],
        lm_weight: Optional[float],
        word_score: Optional[float],
        sil_score: Optional[float],
        subwords: bool = False,
    ) -> None:
        assert (
            lmFile and lexicon
        ), f"If KenLM is used, neither the language model nor the lexicon can be empty!"

        self._nbest = 1
        self.frameSize = 0.02
        self._subwords = subwords

        if "<ctc_blank>" in vocabulary:
            self.silence = vocabulary.index("<ctc_blank>")
        else:
            self.silence = vocabulary.index("<s>")
        if "<sep>" in vocabulary:
            self.boundary = vocabulary.index("<sep>")
        elif "|" in vocabulary:
            self.boundary = vocabulary.index("|")
        else:
            self.boundary = vocabulary.index("</s>")

        lexicon = load_words(lexicon)
        self._wordDict = create_word_dict(lexicon)
        lm = KenLM(lmFile, self._wordDict)
        trie = self._initializeTrie(vocabulary, lm, lexicon)

        decoderOpts = LexiconDecoderOptions(
            beam_size=15,
            beam_size_token=len(vocabulary),
            beam_threshold=25.0,
            lm_weight=lm_weight,
            word_score=word_score,
            unk_score=-np.inf,
            sil_score=sil_score,
            log_add=False,
            criterion_type=CriterionType.CTC,
        )

        self._decoder = LexiconDecoder(
            options=decoderOpts,
            trie=trie,
            lm=lm,
            sil_token_idx=self.boundary,
            blank_token_idx=self.silence,
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
        trie = Trie(len(vocabulary), self.boundary)
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
        allWords, allScores, allTimesteps, allWordsFrames = [], [], [], []
        for b in range(B):
            hypothesis = self._decodeLexicon(emissions, b)
            words, wordsFrames, scores, timesteps = self._postProcessHypothesis(
                hypothesis
            )
            allWords.append(words)
            allWordsFrames.append(wordsFrames)
            allScores.append(scores)
            allTimesteps.append(timesteps)
        return _DecodeResult(
            label_sequences=allWords,
            wordsFrames=allWordsFrames,
            scores=allScores,
            timesteps=allTimesteps,
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
        words, wordsFrames, scores, timesteps = [], [], [], []
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
            wordFrames, wordTimestamps = self._getWordTimestamps(result.tokens)
            wordsFrames.append(wordFrames)
            timesteps.append(wordTimestamps)
        return (words, wordsFrames, scores, timesteps)

    def _getWordTimestamps(self, tokenIdxs: List[int]):
        wordFrames = self._getWordsFrames(tokenIdxs)
        timeIntervals = []
        for frame in wordFrames:
            interval = self._getTimeInterval(frame)
            timeIntervals.append(interval)
        return wordFrames, timeIntervals

    def _getWordsFrames(self, tokenIdxs: List[int]) -> List[List[int]]:
        return FrameToWordProcessor(tokenIdxs, self.silence, self.boundary).invoke()

    def _getTimeInterval(self, frames: List[int]) -> Tuple[float, float]:
        return self._extendFramesToBoundaries(
            self._getFrameTime(frames[0]), self._getFrameTime(frames[len(frames) - 1])
        )

    def _getFrameTime(self, frame: int) -> float:
        return frame * self.frameSize

    def _extendFramesToBoundaries(self, begin: int, end: int) -> Tuple[int, int]:
        return (begin, end + self.frameSize)



class FrameToWordProcessor:
    def __init__(self, tokenIdxs, silence, boundary):
        self.silence = silence
        self.boundary = boundary
        self.timesteps = []
        self.wordFrames = []
        self.wordFound = False
        self.letter = ""
        self.prevLetter = boundary
        self.frames = tokenIdxs

    def invoke(self) -> List[List[int]]:
        for i, l in enumerate(self.frames):
            self.letter = l
            self.current = i
            self.processCurrentFrame()
        return self.timesteps

    def processCurrentFrame(self):
        if self.letter == self.silence:
            pass
        elif self.__wordStartsHere():
            self.wordFound = True
            self.wordFrames.append(self.current)
        elif self.__wordContinues():
            self.wordFrames.append(self.current)
        elif self.__wordEndsHere():
            if self.current-2 == self.wordFrames[-1]:
                self.wordFrames.append(self.current-1)
            self.timesteps.append(self.wordFrames)
            self.__correctLastWordBoundaries()
            self.wordFound = False
            self.wordFrames = []
        self.prevLetter = self.letter

    def __wordStartsHere(self):
        return self.letter != self.prevLetter and self.letter != self.boundary

    def __wordContinues(self):
        return self.wordFound and self.letter != self.boundary

    def __wordEndsHere(self):
        return self.wordFound and self.letter == self.boundary

    def __correctLastWordBoundaries(self):
        if len(self.timesteps)<2:
            return
        endOfFirst = self.timesteps[-2][-1]
        beginOfSecond = self.timesteps[-1][0]
        distance = beginOfSecond - endOfFirst - 1
        if distance<=6 and distance>0:
            self.timesteps[-1].insert(0, beginOfSecond - distance)
        elif distance>6:
            self.timesteps[-1].insert(0, beginOfSecond - 6)
