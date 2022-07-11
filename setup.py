#!/usr/bin/env python3

import os
from setuptools import find_packages, setup


def version():
    with open("VERSION") as f:
        version = f.read().strip()
    with open(os.path.join("asr4", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


def readme():
    with open("README.md") as f:
        readme = f.read()
    return readme


def description():
    with open("DESCRIPTION.md") as f:
        description = f.read()
    return description


if __name__ == "__main__":
    setup(
        name="asr4",
        version=version(),
        description=description(),
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Telecommunications Industry",
            "Natural Language :: English",
            "Natural Language :: Spanish",
            "Natural Language :: Portuguese (Brazilian)",
            "License :: Other/Proprietary License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Multimedia :: Sound/Audio :: Speech",
        ],
        long_description=readme(),
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=[],
        packages=find_packages(
            exclude=[
                "tests",
                "tests.*",
            ]
        ),
        test_suite="tests",
        zip_safe=False,
    )
