from typing import Any, Dict, List
from functools import reduce

from pyformatter import PyFormatter as Formatter
from asr4.types.language import Language
import logging

class FormatterFactory:
    @staticmethod
    def createFormatter(model_path: str, language: Language) -> Formatter:
        return Formatter(
            FormatterFactory._sanitizeLanguage(language.asFormatter()),
            model_path,
            b"",
            b"",
            dict(),
        )

    def _sanitizeLanguage(language: str) -> str:
        if len(language) == 2:
            return language + "-" + language
        return language



class TimeFixer:
    def __init__(self, operations: List[Dict[str,Any]], timestamps: List[Any], frames: List[List[int]]):
        self.operations = operations
        self.timestamps = timestamps
        self.frames = frames
        self.logger = logging.getLogger("ASR4")

    def invoke(self):
        self.logger.debug(f"Fixing timestamps for {len(self.timestamps)} in {len(self.operations)} operations")
        if len(self.frames)==len(self.timestamps):
            for op in self.operations:
                self.runOperation(op)
        else:
            self.logger.warning(
                f"Uneven timestamp data: {len(self.timestamps)} timestamps for {len(self.frames)} frames"
            )
        return (self.timestamps, self.frames)

    def runOperation(self, operation: Dict[str, Any]):
        opType, arg = self.getType(operation)
        if opType == "merge":
            self.mergeIntervalOfWords(arg["start"], arg["end"])
        elif opType == "pass":
            pass
            
    def getType(self, operation: Dict):
        if "Merge" in operation:
            return ("merge", operation["Merge"][0])
        elif "Change" in operation:
            return ("change", operation["Change"][0])
        elif "InsertAfter" in operation:
            return ("pass", None)
        elif "InsertBefore" in operation:
            return ("pass", None)
        
    def mergeIntervalOfWords(self, begin, end):
        if end >= len(self.timestamps) or begin >= len(self.timestamps) or begin<0:
            return
        newBegin = self.timestamps[begin][0]
        newEnd = self.timestamps[end][1]
        self.timestamps[begin:end+1] = [(newBegin,newEnd)]
        self.frames[begin:end+1] = [reduce(lambda a, b: a + b, self.frames[begin:end+1])]

        
