# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'cyclo_lab' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch>=2.10.0",
    "torchvision>=0.25.0",
    # 5.26.0 introduced a breaking change, so we restricted it for now.
    # See issue https://github.com/tensorflow/tensorboard/issues/6808 for details.
    "protobuf>=4.25.8, != 5.26.0",
    # basic logger
    "tensorboard",
    # automate
    "scikit-learn",
    "pynput",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Installation operation
setup(
    name="cyclo_lab",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
