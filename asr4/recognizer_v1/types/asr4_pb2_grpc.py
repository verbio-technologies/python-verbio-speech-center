# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import asr4.recognizer_v1.types.asr4_pb2 as asr4__pb2


class RecognizerStub(object):
    """Service that implements ASR4 Recognition API.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Recognize = channel.unary_unary(
                '/asr4.recognizer.v1.Recognizer/Recognize',
                request_serializer=asr4__pb2.RecognizeRequest.SerializeToString,
                response_deserializer=asr4__pb2.RecognizeResponse.FromString,
                )


class RecognizerServicer(object):
    """Service that implements ASR4 Recognition API.
    """

    def Recognize(self, request, context):
        """Send audio as bytes and recieve the transcription of the audio.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RecognizerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Recognize': grpc.unary_unary_rpc_method_handler(
                    servicer.Recognize,
                    request_deserializer=asr4__pb2.RecognizeRequest.FromString,
                    response_serializer=asr4__pb2.RecognizeResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'asr4.recognizer.v1.Recognizer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Recognizer(object):
    """Service that implements ASR4 Recognition API.
    """

    @staticmethod
    def Recognize(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/asr4.recognizer.v1.Recognizer/Recognize',
            asr4__pb2.RecognizeRequest.SerializeToString,
            asr4__pb2.RecognizeResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
