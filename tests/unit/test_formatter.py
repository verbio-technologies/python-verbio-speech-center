import pytest, unittest
import os, json
from pathlib import Path

#from pyformatter import PyRewritting as Rewritting
from asr4.recognizer_v1.formatter import *
from asr4.recognizer_v1.runtime.onnx import OnnxRuntimeResult
from asr4.types.language import Language

class TestTimestampManagement(unittest.TestCase):
    def testFormat(self):
        text = OnnxRuntimeResult(sequence="something", score=1.0, wordTimestamps= [])
        ops = '[Change(9, "."), Merge(6..=10, "vrbio.com"), Change(5, "@"), Merge(3..=4, "lformiga"), Merge(3..=5, "lformiga@vrbio.com"), Change(0, "My"), Change(3, "lformiga@vrbio.com.")]'

        x = fixTimestamps(text, ops)

class TestFormatter(unittest.TestCase):
    def testFormatter(self):
        w = ["my", "email", "is", "l", "formiga", "at", "v", "r", "bio", "dot", "com"]
        f = FormatterFactory.createFormatter("formatter/format-model.en-us-1.0.1.fm", Language.EN_US)
        (w,ops) = f.classify(w)
        print(ops)
        print(ops.__dict__)
        for op in ops.content:
            print(type(op),op)
