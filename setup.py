#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="immich-static",
    version="0.1.0",
    packages=find_packages(include=["immich_static", "immich_static.*"]),
    install_requires=[
        "opencv-python-headless>=4.5.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "immich-static=immich_static.cli:main",
        ],
    },
)
