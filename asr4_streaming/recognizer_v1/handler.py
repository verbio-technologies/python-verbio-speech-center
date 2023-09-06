import grpc
import logging
import numpy as np
from typing import List
from datetime import timedelta
from dataclasses import dataclass

from .types import Duration
from .types import WordInfo
from .types import SampleRate
from .types import AudioEncoding
from .types import RecognitionConfig
from .types import RecognitionResource
from .types import RecognitionParameters
from .types import RecognitionAlternative
from .types import StreamingRecognizeRequest
from .types import StreamingRecognitionResult

from asr4_engine.data_classes import Signal
from asr4.engines.wav2vec.v1.engine_types import Language
from asr4.engines.wav2vec.wav2vec_engine import Wav2VecEngine
from asr4_engine.data_classes.transcription import Segment, WordTiming


@dataclass
class TranscriptionResult:
    transcription: str
    score: float
    duration: Duration
    words: List[WordTiming]


class EventHandler:
    def __init__(
        self,
        language: Language,
        engine: Wav2VecEngine,
        context: grpc.aio.ServicerContext,
    ):
        self._engine = engine
        self._context = context
        self._language = language
        self._audio = bytearray()
        self._config = RecognitionConfig()
        self._totalDuration = Duration()
        self._logger = logging.getLogger("ASR4")

    async def source(self, request: StreamingRecognizeRequest):
        if request.HasField("config"):
            await self.__validateRecognitionConfig(request.config)
            self._logger.info(
                "Received streaming request "
                f"[language={request.config.parameters.language}] "
                f"[sample_rate={request.config.parameters.sample_rate_hz}] "
                f"[formatting={request.config.parameters.enable_formatting}] "
                f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
            )
            self._config.CopyFrom(request.config)
            self._audio = bytearray()
        elif request.HasField("audio"):
            await self.__validateAudio(request.audio)
            self._logger.info(f"Received partial audio [length={len(request.audio)}]")
            self._audio += request.audio
        else:
            await self.__logError("Empty request", grpc.StatusCode.INVALID_ARGUMENT)

    async def __validateRecognitionConfig(self, config: RecognitionConfig):
        await self.__validateParameters(config.parameters)
        await self.__validateResource(config.resource)

    async def __validateParameters(self, parameters: RecognitionParameters):
        message = ""
        if not Language.check(parameters.language):
            message = f"Invalid value '{parameters.language}' for language parameter"
        if not SampleRate.check(parameters.sample_rate_hz):
            message = f"Invalid value '{parameters.sample_rate_hz}' for sample_rate_hz parameter"
        if not AudioEncoding.check(parameters.audio_encoding):
            message = f"Invalid value '{parameters.audio_encoding}' for audio_encoding parameter"
        if message:
            await self.__logError(message, grpc.StatusCode.INVALID_ARGUMENT)

    async def __validateResource(self, resource: RecognitionResource):
        try:
            RecognitionResource.Model.Name(resource.topic)
        except:
            message = f"Invalid value '{resource.topic}' for topic resource"
            await self.__logError(message, grpc.StatusCode.INVALID_ARGUMENT)

    async def __validateAudio(self, audio: bytes):
        if len(audio) == 0:
            await self.__logError(
                "Empty value for audio", grpc.StatusCode.INVALID_ARGUMENT
            )

    async def handle(self) -> TranscriptionResult:
        await self.__checkIfRecognitionIsPossible()
        duration = self.__getAudioDuration()
        language = Language.parse(self._config.parameters.language)
        if language == self._language:
            return self.__handle(duration)
        else:
            message = f"Invalid language '{language.value}'. Only '{self._language.value}' is supported."
            await self.__logError(message, grpc.StatusCode.INVALID_ARGUMENT)

    async def __checkIfRecognitionIsPossible(self):
        try:
            await self.__validateRecognitionConfig(self._config)
        except:
            message = "RecognitionConfig was never received"
            await self.__logError(message, grpc.StatusCode.INVALID_ARGUMENT)
        try:
            await self.__validateAudio(self._audio)
        except:
            message = "Audio was never received"
            await self.__logError(message, grpc.StatusCode.INVALID_ARGUMENT)

    def __getAudioDuration(self) -> Duration:
        duration = EventHandler.__calculateAudioDuration(self._config, self._audio)
        self._logger.info(
            f"Received total audio [length={len(self._audio)}] "
            f"[duration={duration.ToTimedelta().total_seconds()}] "
        )
        self._totalDuration = EventHandler.__addAudioDuration(
            self._totalDuration, duration
        )
        return duration

    @staticmethod
    def __addAudioDuration(a: Duration, b: Duration) -> Duration:
        duration = Duration()
        total = a.ToTimedelta().total_seconds() + b.ToTimedelta().total_seconds()
        duration.FromTimedelta(td=timedelta(seconds=total))
        return duration

    def __handle(self, duration: Duration) -> TranscriptionResult:
        sampleRate = self._config.parameters.sample_rate_hz
        enableFormatter = self._config.parameters.enable_formatting
        result = self._engine.recognize(
            Signal(np.frombuffer(self._audio, dtype=np.int16), sampleRate),
            language=self._language.value,
            formatter=enableFormatter,
        )
        return TranscriptionResult(
            duration=duration,
            transcription=result.text,
            score=EventHandler.__calculateAverageScore(result.segments),
            words=EventHandler.__extractWords(result.segments),
        )

    @staticmethod
    def __calculateAudioDuration(config: RecognitionConfig, audio: bytes) -> Duration:
        duration = Duration()
        sampleSizeInBytes = AudioEncoding.parse(
            config.parameters.audio_encoding
        ).getSampleSizeInBytes()
        samplesNumber = len(audio) / sampleSizeInBytes
        duration.FromTimedelta(
            td=timedelta(seconds=(samplesNumber / config.parameters.sample_rate_hz))
        )
        return duration

    @staticmethod
    def __calculateAverageScore(segments: List[Segment]) -> float:
        if not len(segments):
            return 0.0
        return sum(s.avg_logprob for s in segments) / len(segments)

    @staticmethod
    def __extractWords(segments: List[Segment]) -> List[WordTiming]:
        return [word for s in segments for word in s.words]

    def sink(
        self,
        response: TranscriptionResult,
    ) -> StreamingRecognitionResult:
        words = [EventHandler.__getWord(word) for word in response.words]
        alternative = RecognitionAlternative(
            transcript=response.transcription, confidence=response.score, words=words
        )
        return StreamingRecognitionResult(
            alternatives=[alternative],
            end_time=self._totalDuration,
            duration=response.duration,
            is_final=True,
        )

    @staticmethod
    def __getWord(word: WordTiming) -> WordInfo:
        wordInfo = WordInfo(
            start_time=Duration(),
            end_time=Duration(),
            word=word.word,
            confidence=word.probability,
        )
        wordInfo.start_time.FromTimedelta(td=timedelta(seconds=word.start))
        wordInfo.end_time.FromTimedelta(td=timedelta(seconds=word.end))
        return wordInfo

    async def __logError(self, message: str, statusCode: grpc.StatusCode):
        self._logger.error(message)
        await self._context.abort(statusCode, message)
