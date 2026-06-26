# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
CYCLO_LAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

CYCLO_LAB_ASSETS_DATA_DIR = os.path.join(CYCLO_LAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

CYCLO_LAB_ASSETS_METADATA = toml.load(os.path.join(CYCLO_LAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = CYCLO_LAB_ASSETS_METADATA["package"]["version"]

from .OMY import *
from .FFW_BG2 import *
from .FFW_SG2 import *
from .FFW_SH5 import *
from .K1_rev1 import *
