# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

from copy import copy
import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union, List
from absl import logging
from yaml import parse
from lib.tool.alphafold.common import residue_constants
from lib.tool.alphafold.data import msa_identifiers
from lib.tool.alphafold.data import parsers
from lib.tool.alphafold.data import templates
from lib.tool.alphafold.data.tools import hhblits
from lib.tool.alphafold.data.tools import hhsearch
from lib.tool.alphafold.data.tools import hmmsearch
from lib.tool.alphafold.data.tools import jackhmmer
from lib.tool.alphafold.model import features

import numpy as np


# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Constructs a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return features


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence.")
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
            )
            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            identifiers = msa_identifiers.get_identifiers(
                msa.descriptions[sequence_index]
            )
            species_ids.append(identifiers.species_id.encode("utf-8"))

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)
    features["msa_species_identifiers"] = np.array(species_ids, dtype=np.object_)
    return features


def load_a3m_with_limited_depth(a3m_file, depth):
    count = 0
    lines = []
    with open(a3m_file, "r") as fd:
        for line in fd:
            if not line.startswith(">"):
                count += 1
            lines.append(line)
            if count >= depth:
                break
    return "".join(lines)


def load_msa_from_path(
    msa_path: str, msa_format: str, max_depth: Optional[int] = None
) -> Mapping[str, Any]:
    """Runs an MSA tool, checking if output already exists first."""
    logging.warning("Reading MSA from file %s", msa_path)
    if msa_format == "sto" and max_depth is not None:
        precomputed_msa = parsers.truncate_stockholm_msa(msa_path, max_depth)
        result = {"sto": precomputed_msa}
    elif msa_format == "a3m" and max_depth is not None:
        result = {msa_format: load_a3m_with_limited_depth(msa_path, depth=max_depth)}
    else:
        with open(msa_path, "r") as f:
            result = {msa_format: f.read()}
    return result


class TemplateProcessor:
    """Runs the alignment tools and assembles the input features."""

    def __init__(
        self,
        template_searching_msa_path: str,
        template_searcher: TemplateSearcher,
        template_featurizer: templates.TemplateHitFeaturizer,
    ):
        """Initializes the data pipeline."""
        self.template_searching_msa_path = template_searching_msa_path
        self.template_searcher = template_searcher
        self.template_featurizer = template_featurizer
        self.format = "a3m"

    def process(self, input_sequence, input_description) -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""


class MonomerMSAFeatureProcessor:
    """Runs the alignment tools and assembles the input features."""

    def __init__(self, msa_paths: dict):
        """Initializes the data pipeline."""
        self.msa_paths = copy(msa_paths)
        self.format = "a3m"

    def process(self, input_sequence, input_description) -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""

        num_res = len(input_sequence)

        msa_collections = []
        for msa_path in self.msa_paths:
            _msa = load_msa_from_path(msa_path, self.format)
            _name = os.path.basename(msa_path)
            msa_collections.append((_name, _msa))

        msas = [
            (_name, parsers.parse_a3m(_msa[self.format]))
            for _name, _msa in msa_collections
        ]

        sequence_features = make_sequence_features(
            sequence=input_sequence, description=input_description, num_res=num_res
        )

        msa_features = make_msa_features(msas=[_m for _n, _m in msas])
        for _name, _msa in msas:
            logging.info(f"{_name} MSA size: {len(_msa)} sequences.")

        logging.info(
            f'Final (deduplicated) MSA size: {msa_features["num_alignments"][0]} sequences.'
        )

        return {**sequence_features, **msa_features}
