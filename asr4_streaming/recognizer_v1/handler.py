import grpc
import soxr
import logging
import numpy as np
from asyncio import Event
from datetime import timedelta
from dataclasses import dataclass
from typing import List, Optional

from .types import Duration
from .types import WordInfo
from .types import SampleRate
from .types import AudioEncoding
from .types import RecognitionConfig
from .types import RecognitionResource
from .types import RecognitionParameters
from .types import RecognitionAlternative
from .types import StreamingRecognizeRequest
from .types import StreamingRecognizeResponse
from .types import StreamingRecognitionResult

from asr4_engine.data_classes import Signal
from asr4.engines.wav2vec.v1.engine_types import Language
from asr4_engine.data_classes.transcription import Segment, WordTiming
from asr4.engines.wav2vec.wav2vec_engine import (
    Wav2VecEngine,
    Wav2VecASR4EngineOnlineHandler,
)


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
        self._config = RecognitionConfig()
        self._totalDuration = 0.0
        self._logger = logging.getLogger("ASR4")
        self._startListening = Event()
        self._onlineHandler: Optional[Wav2VecASR4EngineOnlineHandler] = None

    async def notifyEndOfAudio(self):
        self._startListening.set()
        if self._onlineHandler:
            await self._onlineHandler.notifyEndOfAudio()

    async def source(self, request: StreamingRecognizeRequest):
        if request.HasField("config"):
            await self.__sourceConfig(request.config)
        elif request.HasField("audio"):
            await self.__sourceAudio(request.audio)
        else:
            await self.__logError("Empty request", grpc.StatusCode.INVALID_ARGUMENT)

    async def __sourceConfig(self, config: RecognitionConfig):
        await self.__validateRecognitionConfig(config)
        self._logger.info(
            "Received streaming request "
            f"[language={config.parameters.language}] "
            f"[sample_rate={config.parameters.sample_rate_hz}] "
            f"[formatting={config.parameters.enable_formatting}] "
            f"[topic={RecognitionResource.Model.Name(config.resource.topic)}]"
        )
        self._config.CopyFrom(config)
        self._onlineHandler = self._engine.getRecognizerHandler(
            language=self._config.parameters.language,
            formatter=self._config.parameters.enable_formatting,
        )
        self._startListening.set()

    async def __validateRecognitionConfig(self, config: RecognitionConfig):
        await self.__validateParameters(config.parameters)
        await self.__validateResource(config.resource)

    async def __validateParameters(self, parameters: RecognitionParameters):
        message = ""
        if not Language.check(parameters.language):
            message = f"Invalid value '{parameters.language}' for language parameter"
        if Language.parse(parameters.language) != self._language:
            message = f"Invalid language '{parameters.language.value}'. Only '{self._language.value}' is supported."
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

    async def __sourceAudio(self, audio: bytes):
        await self.__validateAudio(audio)
        self._logger.info(f"Received partial audio [length={len(audio)}]")
        if self._onlineHandler:
            await self._onlineHandler.sendAudioChunk(
                self.__convertAudioToSignal(
                    audio=audio, sampleRate=self._config.parameters.sample_rate_hz
                )
            )
        else:
            await self.__logError(
                "A request containing RecognitionConfig must be sent first",
                grpc.StatusCode.INVALID_ARGUMENT,
            )

    async def __validateAudio(self, audio: bytes):
        if len(audio) == 0:
            await self.__logError(
                "Empty value for audio", grpc.StatusCode.INVALID_ARGUMENT
            )

    def __convertAudioToSignal(self, audio: bytes, sampleRate: int) -> Signal:
        audioArray = np.frombuffer(audio, dtype=np.int16)
        audioArrayResampled = soxr.resample(audioArray, sampleRate, 16000)
        return Signal(audioArrayResampled, 16000)

    async def listenForTranscription(self):
        await self._startListening.wait()
        if self._onlineHandler:
            async for partialResult in self._onlineHandler.listenForCompleteAudio():
                self._logger.info(f"Partial recognition result: '{partialResult.text}'")
                partialTranscriptionResult = TranscriptionResult(
                    transcription=partialResult.text,
                    duration=self.__getDuration(partialResult.duration),
                    score=EventHandler.__calculateAverageScore(partialResult.segments),
                    words=EventHandler.__extractWords(partialResult.segments),
                )
                await self._context.write(self.sink(partialTranscriptionResult))
                self._totalDuration += partialResult.duration

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
    ) -> StreamingRecognizeResponse:
        words = [
            EventHandler.__getWord(word, self._totalDuration) for word in response.words
        ]
        alternative = RecognitionAlternative(
            transcript=response.transcription, confidence=response.score, words=words
        )
        return StreamingRecognizeResponse(
            results=StreamingRecognitionResult(
                alternatives=[alternative],
                end_time=EventHandler.__addAudioDuration(
                    EventHandler.__getDuration(self._totalDuration), response.duration
                ),
                duration=response.duration,
                is_final=True,
            )
        )

    @staticmethod
    def __getWord(word: WordTiming, timeOffset: float) -> WordInfo:
        return WordInfo(
            start_time=EventHandler.__getDuration(word.start + timeOffset),
            end_time=EventHandler.__getDuration(word.end + timeOffset),
            word=word.word,
            confidence=word.probability,
        )

    @staticmethod
    def __getDuration(seconds: float) -> Duration:
        duration = Duration()
        duration.FromTimedelta(td=timedelta(seconds=seconds))
        return duration

    @staticmethod
    def __addAudioDuration(a: Duration, b: Duration) -> Duration:
        duration = Duration()
        total = a.ToTimedelta().total_seconds() + b.ToTimedelta().total_seconds()
        duration.FromTimedelta(td=timedelta(seconds=total))
        return duration

    async def __logError(self, message: str, statusCode: grpc.StatusCode):
        self._logger.error(message)
        await self._context.abort(statusCode, message)
