import os

COMPILED_GRAMMAR_FORMAT = '.tar'
COMPILED_GRAMMAR_SUB_FORMAT = '.xz'


def check_format(compiled_grammar: str):
    first_split = os.path.splitext(compiled_grammar)
    second_split = os.path.splitext(first_split[0])
    return first_split[1] == COMPILED_GRAMMAR_SUB_FORMAT and second_split[1] == COMPILED_GRAMMAR_FORMAT


def get_compiled_grammar(compiled_grammar: str):
    if not os.path.exists(compiled_grammar):
        raise ValueError(f"{compiled_grammar} file does not exist.")
    if not check_format(compiled_grammar):
        raise ValueError(f"{compiled_grammar} file specified is not {COMPILED_GRAMMAR_FORMAT}"
                         f"{COMPILED_GRAMMAR_SUB_FORMAT}.")

    with open(compiled_grammar, mode="rb") as f:
        data = f.read()
    return data
