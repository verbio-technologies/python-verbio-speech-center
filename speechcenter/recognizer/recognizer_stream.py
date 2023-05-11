#!/usr/bin/env python3
# Used to make sure python find proto files
import threading
from typing import Iterator, Iterable
import logging
from grpc import Channel
from authentication import retrieveToken
from options import Options
from audio import Audio
from proto.generated import recognition_pb2_grpc as recognitionGrpc
from proto.generated import recognition_streaming_request_pb2 as streamingRequestProto
from proto.generated import recognition_streaming_response_pb2 as streamingResponseProto

from google.protobuf.json_format import MessageToJson


class InactivityTimer:
    def __init__(self, timeout: float, closeInactiveStream):
        self.timer = None
        self.timeout = timeout
        self.closeInactiveStream = closeInactiveStream
    
    def start(self):
        self.timer = threading.Timer(self.timeout, self.closeInactiveStream)
        self.timer.start()
    
    def running(self):
        return bool(self.timer)
    
    def cancel(self):
        if self.timer:
            self.timer.cancel()


class SpeechCenterStreamingASRClient:
    def __init__(self, channel: Channel, options: Options):
        self.channel = channel
        self.stub = recognitionGrpc.RecognizerStub(self.channel)
        self.audio = Audio(options)
        self.options = options
        self.token = retrieveToken(options)
        self.peer_responded = threading.Event()
        self.inactivity_timer = InactivityTimer(options.inactivity_timeout, self.closeStreamByInactivity)
    
    @staticmethod
    def buildConfigRequest(options: Options, audio: Audio) -> streamingRequestProto.RecognitionStreamingRequest:
        return streamingRequestProto.RecognitionStreamingRequest(
                config=streamingRequestProto.RecognitionConfig(
                    parameters=streamingRequestProto.RecognitionParameters(
                        language=options.language,
                        pcm=streamingRequestProto.PCM(sample_rate_hz=audio.sample_rate),
                        enable_formatting = options.formatting,
                        enable_diarization = options.diarization
                    ), 
                    resource=streamingRequestProto.RecognitionResource(topic=options.topic),
                    label=[options.label],
                    version=options.asr_version
                )
            )
    
    @staticmethod
    def buildAudioRequests(audio: Audio) -> Iterable[streamingRequestProto.RecognitionStreamingRequest]:
        count = 0
        for chunk in audio.divide():
            count += 1
            logging.info("Sending audio streaming message #" + str(count))
            logging.debug(repr(chunk)[0:50] + "...")
            yield streamingRequestProto.RecognitionStreamingRequest(audio=chunk)

        logging.info("All audio messages sent")
    
    def sendMessages(self) -> Iterable[streamingRequestProto.RecognitionStreamingRequest]:
        yield self.buildConfigRequest(self.options, self.audio)

        for audio_request in self.buildAudioRequests(self.audio):
            yield audio_request

    def closeStreamByInactivity(self):
        logging.info("Stream inactivity detected, closing stream...")
        self.peer_responded.set()

    def _printResult(self, response):
        duration = response.result.duration
        for alternative in response.result.alternatives:
            if alternative.transcript:
                print('\t"transcript": "%s",\n\t"confidence": "%f",\n\t"duration": "%f"' % (alternative.transcript, alternative.confidence, duration))

    def _responseWatcher(
            self,
            response_iterator: Iterator[streamingResponseProto.RecognitionStreamingResponse]) -> None:
        try:
            logging.info("Running response watcher")
            for response in response_iterator:
                json = MessageToJson(response)
                logging.info("New incoming response: '%s ...'", json[0:50].replace('\n', ''))
                self._printResult(response)

                if response.result and response.result.is_final:
                    if self.inactivity_timer.running():
                        self.inactivity_timer.cancel()
                    self.inactivity_timer.start()

        except Exception as e:
            logging.error("Error running response watcher: %s", str(e))
            self.peer_responded.set()
            raise
    
    def call(self) -> None:
        metadata = [('authorization', "Bearer " + self.token)]
        if self.secure_channel:
            response_iterator = self._stub.StreamingRecognize(self.sendMessages())
        else:
            response_iterator = self._stub.StreamingRecognize(self.sendMessages(), metadata=metadata)
        
        self._consumer_future = self._executor.submit(self._responseWatcher, response_iterator)
    
    def wait_server(self) -> bool:
        logging.info("Waiting for server to respond...")
        self.peer_responded.wait(timeout=None)
        if self._consumer_future.done():
            # If the future raises, forwards the exception here
            self._consumer_future.result()
        
        return True
 