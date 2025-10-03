# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch


from convertPY.main.utils_main import write_ply, load_config
from convertUTILS.rich_utils import CONSOLE

@dataclass
class BaseCull:
    """Base class for rendering."""
    model_path: Path
    """Path to model file."""
    output_dir: Path = Path("culled_models/output.ply")
    """Path to output model file."""


@dataclass
class Convert(BaseCull):
    """Cull using all images in the dataset."""

    mask_dir: Optional[Path] = None

    def run_convert(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config, pipeline = load_config(
            self.model_path,
            test_mode="inference",
        )
        config.datamanager.dataparser.downscale_factor = 1

        CONSOLE.log("Writing to ply...")
        filename = write_ply(self.model_path, pipeline.model)
        path = Path(filename)
        print(f"PLY FILE: {path}")
