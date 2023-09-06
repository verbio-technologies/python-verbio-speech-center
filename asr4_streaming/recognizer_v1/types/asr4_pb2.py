# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: asr4.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2
from google.rpc import status_pb2 as google_dot_rpc_dot_status__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\nasr4.proto\x12\x12\x61sr4.recognizer.v1\x1a\x1egoogle/protobuf/duration.proto\x1a\x17google/rpc/status.proto"z\n\x19StreamingRecognizeRequest\x12\x37\n\x06\x63onfig\x18\x01 \x01(\x0b\x32%.asr4.recognizer.v1.RecognitionConfigH\x00\x12\x0f\n\x05\x61udio\x18\x02 \x01(\x0cH\x00\x42\x13\n\x11streaming_request"\x8d\x01\n\x11RecognitionConfig\x12=\n\nparameters\x18\x01 \x01(\x0b\x32).asr4.recognizer.v1.RecognitionParameters\x12\x39\n\x08resource\x18\x02 \x01(\x0b\x32\'.asr4.recognizer.v1.RecognitionResource"\xc7\x01\n\x15RecognitionParameters\x12\x10\n\x08language\x18\x01 \x01(\t\x12\x16\n\x0esample_rate_hz\x18\x02 \x01(\r\x12O\n\x0e\x61udio_encoding\x18\x03 \x01(\x0e\x32\x37.asr4.recognizer.v1.RecognitionParameters.AudioEncoding\x12\x19\n\x11\x65nable_formatting\x18\x04 \x01(\x08"\x18\n\rAudioEncoding\x12\x07\n\x03PCM\x10\x00"i\n\x13RecognitionResource\x12<\n\x05topic\x18\x01 \x01(\x0e\x32-.asr4.recognizer.v1.RecognitionResource.Model"\x14\n\x05Model\x12\x0b\n\x07GENERIC\x10\x00"\x9a\x01\n\x1aStreamingRecognizeResponse\x12#\n\x05\x65rror\x18\x01 \x01(\x0b\x32\x12.google.rpc.StatusH\x00\x12\x41\n\x07results\x18\x02 \x01(\x0b\x32..asr4.recognizer.v1.StreamingRecognitionResultH\x00\x42\x14\n\x12streaming_response"\xca\x01\n\x1aStreamingRecognitionResult\x12@\n\x0c\x61lternatives\x18\x01 \x03(\x0b\x32*.asr4.recognizer.v1.RecognitionAlternative\x12+\n\x08\x65nd_time\x18\x02 \x01(\x0b\x32\x19.google.protobuf.Duration\x12\x10\n\x08is_final\x18\x03 \x01(\x08\x12+\n\x08\x64uration\x18\x04 \x01(\x0b\x32\x19.google.protobuf.Duration"m\n\x16RecognitionAlternative\x12\x12\n\ntranscript\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12+\n\x05words\x18\x03 \x03(\x0b\x32\x1c.asr4.recognizer.v1.WordInfo"\x88\x01\n\x08WordInfo\x12-\n\nstart_time\x18\x01 \x01(\x0b\x32\x19.google.protobuf.Duration\x12+\n\x08\x65nd_time\x18\x02 \x01(\x0b\x32\x19.google.protobuf.Duration\x12\x0c\n\x04word\x18\x03 \x01(\t\x12\x12\n\nconfidence\x18\x04 \x01(\x02\x32\x85\x01\n\nRecognizer\x12w\n\x12StreamingRecognize\x12-.asr4.recognizer.v1.StreamingRecognizeRequest\x1a..asr4.recognizer.v1.StreamingRecognizeResponse(\x01\x30\x01\x62\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "asr4_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _STREAMINGRECOGNIZEREQUEST._serialized_start = 91
    _STREAMINGRECOGNIZEREQUEST._serialized_end = 213
    _RECOGNITIONCONFIG._serialized_start = 216
    _RECOGNITIONCONFIG._serialized_end = 357
    _RECOGNITIONPARAMETERS._serialized_start = 360
    _RECOGNITIONPARAMETERS._serialized_end = 559
    _RECOGNITIONPARAMETERS_AUDIOENCODING._serialized_start = 535
    _RECOGNITIONPARAMETERS_AUDIOENCODING._serialized_end = 559
    _RECOGNITIONRESOURCE._serialized_start = 561
    _RECOGNITIONRESOURCE._serialized_end = 666
    _RECOGNITIONRESOURCE_MODEL._serialized_start = 646
    _RECOGNITIONRESOURCE_MODEL._serialized_end = 666
    _STREAMINGRECOGNIZERESPONSE._serialized_start = 669
    _STREAMINGRECOGNIZERESPONSE._serialized_end = 823
    _STREAMINGRECOGNITIONRESULT._serialized_start = 826
    _STREAMINGRECOGNITIONRESULT._serialized_end = 1028
    _RECOGNITIONALTERNATIVE._serialized_start = 1030
    _RECOGNITIONALTERNATIVE._serialized_end = 1139
    _WORDINFO._serialized_start = 1142
    _WORDINFO._serialized_end = 1278
    _RECOGNIZER._serialized_start = 1281
    _RECOGNIZER._serialized_end = 1414
# @@protoc_insertion_point(module_scope)
