# Copyright 2021 Adap GmbH. All Rights Reserved.
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
# ==============================================================================
"""Creates a PyTorch Dataset for Leaf Shakespeare."""
from __future__ import print_function
import warnings
import os
import os.path
import csv

from typing import Tuple
import pickle
from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import Dataset


LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


class SHAKESPEARE():
    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, dataset='train', transform=None, target_transform=None):

        self.characters = LEAF_CHARACTERS
        self.num_letters = len(self.characters)  # 80

        self.data_file = dataset  # 'train', 'test', 'validation'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(self.processed_folder, self.data_file)

        # load data and targets
        self.data, self.targets = self.load_file(self.path)
        # self.mapping = {idx:file for idx, file in enumerate(raw_data)}

    def word_to_indices(self, word):
        """Converts a sequence of characters into position indices in the
        reference string `self.characters`.
        Args:
            word (str): Sequence of characters to be converted.
        Returns:
            List[int]: List with positions.
        """
        indices = [self.characters.find(c) for c in word]
        return indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        data = self.data[index]
        pickle_name = data[0]
        idx = int(data[1])

        with open(pickle_name, 'rb') as f:
            d = pickle.load(f)
            x = d['x'][idx]
            y = d['y'][idx]

        sentence_indices = np.array(self.word_to_indices(x))
        next_word_index = np.array(self.characters.find(y))

        return sentence_indices, next_word_index

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, path):
        datas, labels = [], []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    datas.append((row[1], row[2]))
                    labels.append(row[-1])
                line_count += 1

        return datas, labels

    def load_file(self, path):

        # load meta file to get labels
        datas, labels = self.load_meta_data(
            os.path.join(self.processed_folder, 'client_data_mapping', self.data_file + '.csv'))

        return datas, labels
