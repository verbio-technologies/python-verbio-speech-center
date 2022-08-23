import abc
from typing import NamedTuple


class Runtime(abc.ABC):
    def run(self, _input: bytes) -> NamedTuple:
        raise NotImplementedError()
