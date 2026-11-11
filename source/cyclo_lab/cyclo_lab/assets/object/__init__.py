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
CYCLO_LAB_OBJECT_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

CYCLO_LAB_OBJECT_ASSETS_DATA_DIR = os.path.join(CYCLO_LAB_OBJECT_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

CYCLO_LAB_OBJECT_ASSETS_METADATA = toml.load(os.path.join(CYCLO_LAB_OBJECT_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = CYCLO_LAB_OBJECT_ASSETS_METADATA["package"]["version"]

from .brush_ring import *
from .jelly_bag import *
from .peanut_mix_bag import *
from .plastic_basket import *
from .plastic_basket2 import *
from .plastic_bottle import *
from .pliers_ring import *
from .roasted_chestnut_bag import *
from .robotis_aiworker_table import *
from .robotis_net_table import *
from .robotis_omy_table import *
from .scissors_ring import *
from .screw_driver_ring import *
from .silicone_tube_ring import *
from .tooth_brush import *
