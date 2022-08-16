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
        author="Verbio Technologies S.L.",
        author_email="squad2@verbio.com",
        url="www.verbio.com",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Telecommunications Industry",
            "Natural Language :: English",
            "Natural Language :: Spanish",
            "Natural Language :: Portuguese (Brazilian)",
            "License :: Other/Proprietary License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Multimedia :: Sound/Audio :: Speech",
        ],
        long_description=readme(),
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=[
            "google > =3.0.0",
            "grpcio >= 1.47.0",
            "logger >= 1.4",
            "protobuf >= 3.20.1",
            "typing_extensions >= 4.3.0",
        ],
        packages=find_packages(
            exclude=[
                "tests",
                "tests.*",
            ]
        ),
        python_requires=">=3.6",
        test_suite="tests",
        zip_safe=False,
    )
