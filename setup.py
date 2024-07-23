#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="bansuri_tts",
    version="0.0.2rc0",
    description="Flow Matching TTS model",
    author="Ashwin Sankar",
    author_email="ashwins1211@gmail.com",
    url="https://github.com/iamunr4v31/bansuri-tts",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = bansuri_tts.train:main",
            "eval_command = bansuri_tts.eval:main",
        ]
    },
)
