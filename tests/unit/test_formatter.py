import pytest, unittest
import os, json
from pathlib import Path

from asr4.recognizer_v1.formatter import *
from asr4.recognizer_v1.runtime.onnx import OnnxRuntimeResult
from asr4.types.language import Language


class TestFormatter(unittest.TestCase):
    def testFormatter(self):
        w = ["my", "email", "is", "l", "formiga", "at", "v", "r", "bio", "dot", "com"]
        f = FormatterFactory.createFormatter("formatter/format-model.en-us-1.0.1.fm", Language.EN_US)
        (w,ops) = f.classify(w)
        ops = json.loads(ops.to_json()).get("operations", [])
        self.assertEqual(len(ops), 7)
        self.assertEqual(ops[0], {'Change': [9, '.']})
        self.assertEqual(ops[6], {'Change': [3, 'Lformiga@vrbio.com']})


class TestTimestampManagement(unittest.TestCase):
    def testFormat(self):
        text = OnnxRuntimeResult(sequence="something", score=1.0, wordTimestamps= [])
        ops = json.loads("{\"operations\": [{\"Change\": [9, \".\"]}, {\"Merge\": [{\"start\": 6, \"end\": 10}, \"vrbio.com\"]}, {\"Change\": [5, \"@\"]}, {\"Merge\": [{\"start\": 3, \"end\": 4}, \"lformiga\"]}, {\"Merge\": [{\"start\": 3, \"end\": 5}, \"lformiga@vrbio.com\"]}, {\"Change\": [0, \"My\"]}, {\"Change\": [3, \"Lformiga@vrbio.com\"]}], \"original_len\": 11}")
        x = fixTimestamps(text, ops)



