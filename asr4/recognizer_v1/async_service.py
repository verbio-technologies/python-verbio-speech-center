import grpc
import logging

from .types import RecognizerServicer
from .types import RecognizeRequest
from .types import RecognizeResponse
from .types import RecognitionResource


class RecognizerServiceAsync(RecognizerServicer):
    async def Recognize(
        self,
        request: RecognizeRequest,
        _context: grpc.aio.ServicerContext,
    ) -> RecognizeResponse:
        """
        Send audio as bytes and receive the transcription of the audio.
        """
        logging.info(
            "Received request "
            f"[language={request.config.parameters.language}] "
            f"[sample_rate={request.config.parameters.sample_rate_hz}] "
            f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
        )
        result = {"text": "Hello I am up and running received a message from you"}
        return RecognizeResponse(**result)
