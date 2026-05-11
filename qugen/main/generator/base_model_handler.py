# Copyright 2023 QUTAC, BASF Digital Solutions GmbH, BMW Group, 
# Lufthansa Industry Solutions AS GmbH, Merck KGaA (Darmstadt, Germany), 
# Munich Re, SAP SE.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt


class BaseModelHandler(ABC):
    """
    It implements the interface for each of the models handlers (continuous QGAN/QCBM and discrete QGAN/QCBM),
    which includes building the models, training them, saving and reloading them, and generating samples from them.
    """

    def __init__(self):
        """"""
        self.device_configuration = None

    @abstractmethod
    def build(self, *args, **kwargs) -> "BaseModelHandler":
        """
        Define the architecture of the model. Weights initialization is also typically performed here.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, file_path: Path, overwrite: bool = True) -> "BaseModelHandler":
        """
        Saves the model weights to a file.

        Parameters:
            file_path (pathlib.Path): destination file for model weights
            overwrite (bool): Flag indicating if any existing file at the target location should be overwritten
        """
        raise NotImplementedError

    @abstractmethod
    def reload(self, file_path: Path) -> "BaseModelHandler":
        """
        Loads the model from a set of weights.

        Parameters:
            file_path (pathlib.Path): source file for the model weights
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, *args) -> "BaseModelHandler":
        """
        Perform training of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args) -> np.array:
        """
        Draw samples from the model.
        """
        raise NotImplementedError