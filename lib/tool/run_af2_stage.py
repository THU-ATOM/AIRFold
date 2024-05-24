import os
import time

# import _pickle as pkl
import pickle as pkl
from absl import app
from absl import flags
from absl import logging

from pathlib import Path
from lib.tool.alphafold.data.pipeline import FeatureDict
from lib.tool.alphafold.data.from_msa_to_feature import (
    MonomerMSAFeatureProcessor,
    load_msa_from_path,
)
from lib.tool.alphafold.data import templates
from lib.tool.alphafold.data.tools import hhsearch
from lib.tool.alphafold.data import parsers
from lib.tool.alphafold.model import features
from lib.tool.alphafold.model import config
from typing import List, Any, Tuple, Dict, Union, Sequence
from lib.tool.alphafold.model import data
from lib.tool.alphafold.model import model
from lib.tool.alphafold.relax import relax
from lib.tool.alphafold.common import residue_constants
from lib.tool.alphafold.common import protein
import dataclasses

import numpy as np

logging.set_verbosity(logging.INFO)

flags.DEFINE_enum(
    "run_stage",
    default="monomer_msa2feature",
    enum_values=[
        "search_template",
        "make_template_feature",
        "monomer_msa2feature",
        "predict_structure",
        "run_relaxation",
    ],
    help="specify which stage to run.",
)
flags.DEFINE_string(
    "argument_path",
    None,
    required=True,
    help="specify the file that stores the augment pkl.",
)
flags.DEFINE_string(
    "tmpdir",
    None,
    required=True,
    help="specify the file that will store the tmporary results.",
)

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def search_template(
    input_sequence: str,
    template_searching_msa_path: str,
    pdb70_database_path: str = "/data/protein/alphafold/pdb70/pdb70",
    hhsearch_binary_path="hhsearch",
) -> Sequence[Dict]:
    template_searcher = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path, databases=[pdb70_database_path]
    )
    msa_for_templates = load_msa_from_path(
        template_searching_msa_path, "a3m", max_depth=None
    )
    pdb_templates_result = template_searcher.query(a3m=msa_for_templates["a3m"])
    pdb_template_hits = template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=input_sequence
    )

    return [dataclasses.asdict(h) for h in pdb_template_hits]


def make_template_feature(
    input_sequence: str,
    pdb_template_hits: Sequence[Dict],
    max_template_hits: int = 20,
    template_mmcif_dir: str = "/data/protein/alphafold/pdb_mmcif/mmcif_files",
    max_template_date: str = "2022-05-05",
    obsolete_pdbs_path: str = "/data/protein/alphafold/pdb_mmcif/obsolete.dat",
    kalign_binary_path: str = "kalign",
) -> Dict:

    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_mmcif_dir,
        max_template_date=max_template_date,
        max_hits=max_template_hits,
        kalign_binary_path=kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=obsolete_pdbs_path,
    )
    pdb_template_hits = [parsers.TemplateHit(**h) for h in pdb_template_hits]
    templates_result = template_featurizer.get_templates(
        query_sequence=input_sequence, hits=pdb_template_hits
    )

    logging.info(
        f"Total number of templates (NB: this can include bad "
        f'templates and is later filtered to top 4): {templates_result.features["template_domain_names"].shape[0]}.'
    )

    return {**templates_result.features}


def monomer_msa2feature(
    sequence: str,
    target_name: str,
    msa_paths: List[str],
    template_feature: dict,
    model_name: str = "model_1",
    random_seed: int = 0,
    **kwargs,
) -> Tuple[FeatureDict, Dict]:
    model_config = config.model_config(model_name)
    if "num_ensemble" in kwargs:
        model_config.data.eval.num_ensemble = kwargs["num_ensemble"]
    if "max_recycles" in kwargs:
        model_config.model.num_recycle = kwargs["max_recycles"]
        model_config.data.common.num_recycle = kwargs["max_recycles"]
    if "max_msa_clusters" in kwargs:
        model_config.data.eval.max_msa_clusters = kwargs["max_msa_clusters"]
        model_config.data.common.reduce_msa_clusters_by_max_templates = False
    if "max_extra_msa" in kwargs:
        model_config.data.common.max_extra_msa = kwargs["max_extra_msa"]

    data_pipe = MonomerMSAFeatureProcessor(
        msa_paths=msa_paths,
    )
    timings = {}
    t_0 = time.time()

    raw_features = data_pipe.process(
        input_sequence=sequence, input_description=target_name
    )
    raw_features_with_template = {**raw_features, **template_feature}

    feat = features.np_example_to_features(
        np_example=raw_features_with_template,
        config=model_config,
        random_seed=random_seed,
    )
    t_diff = time.time() - t_0
    timings["monomer_msa2feature"] = t_diff
    return feat, timings


def predict_structure(
    target_name: str,
    processed_feature: FeatureDict,
    model_name: str = "model_1",
    data_dir: str = "/data/protein/alphafold",
    random_seed: int = 0,
    return_representations: bool = True,
    **kwargs,
) -> Tuple[Dict, str, dict]:
    """Predicts structure using AlphaFold for the given sequence."""

    model_config = config.model_config(model_name)

    if "num_ensemble" in kwargs:
        model_config.data.eval.num_ensemble = kwargs["num_ensemble"]
    if "max_recycles" in kwargs:
        model_config.model.num_recycle = kwargs["max_recycles"]
        model_config.data.common.num_recycle = kwargs["max_recycles"]
    if "max_msa_clusters" in kwargs:
        model_config.data.eval.max_msa_clusters = kwargs["max_msa_clusters"]
        model_config.data.common.reduce_msa_clusters_by_max_templates = False
    if "max_extra_msa" in kwargs:
        model_config.data.common.max_extra_msa = kwargs["max_extra_msa"]
    run_multimer_system = kwargs.get("run_multimer_system", False)

    if run_multimer_system:
        model_config.model.num_ensemble_eval = model_config.data.eval.num_ensemble

    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
    model_runner = model.RunModel(
        model_config,
        model_params,
        return_representations=return_representations,
    )

    logging.info("Predicting %s", target_name)
    timings = {}

    logging.info("Running model %s on %s", model_name, target_name)

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature, random_seed=random_seed)
    t_diff = time.time() - t_0
    timings[f"predict_and_compile_{model_name}"] = t_diff
    logging.info(
        "Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs",
        model_name,
        target_name,
        t_diff,
    )
    plddt = prediction_result["plddt"]

    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1
    )
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode,
    )
    unrelaxed_pdb_str = protein.to_pdb(unrelaxed_protein)

    # available keys in representations, we only need structure_module
    save_keys = [
        # "msa",
        # "msa_first_row",
        # "pair",
        # "single",
        "structure_module",
    ]
    if "representations" in prediction_result:
        prediction_result["representations"] = {
            key: np.asarray(val)
            for key, val in prediction_result["representations"].items()
            if key in save_keys
        }
    return prediction_result, unrelaxed_pdb_str, timings


def run_relaxation(
    unrelaxed_pdb_str: str, use_gpu_relax: bool = True, **kwargs
) -> Tuple[protein.Protein, Dict]:

    unrelaxed_protein = protein.from_pdb_string(unrelaxed_pdb_str)

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=use_gpu_relax,
    )
    timings = {}
    t_0 = time.time()
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    t_diff = time.time() - t_0
    logging.info(f"Total relaxation time: {t_diff}")
    timings["relaxation"] = t_diff

    return relaxed_pdb_str, timings


def save_protein_as_pdb_file(prot: protein.Protein, path: Union[str, Path]) -> str:
    pdb_string = protein.to_pdb(prot=prot)
    with open(path, "w") as f:
        f.write(pdb_string)

    return path


def save_object_as_pickle(obj: Any, path: Union[str, Path]):
    with open(path, "wb") as fd:
        pkl.dump(obj=obj, file=fd, protocol=4)
    return path


def main(argv):

    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    with open(FLAGS.argument_path, "rb") as fd:
        argument_dict = pkl.load(fd)
    print("the argument dict is:", [k for k in argument_dict])
    returns_pkl_path = os.path.join(FLAGS.tmpdir, "returns.pkl")

    if FLAGS.run_stage == "search_template":
        pdb_template_hits = search_template(**argument_dict)
        save_object_as_pickle(pdb_template_hits, returns_pkl_path)
    elif FLAGS.run_stage == "make_template_feature":
        template_features = make_template_feature(**argument_dict)
        save_object_as_pickle(template_features, returns_pkl_path)
    elif FLAGS.run_stage == "monomer_msa2feature":
        feats, timings = monomer_msa2feature(**argument_dict)
        save_object_as_pickle((feats, timings), returns_pkl_path)
    elif FLAGS.run_stage == "predict_structure":
        results, unrelaxed_pdb_str, timings = predict_structure(**argument_dict)
        save_object_as_pickle((results, unrelaxed_pdb_str, timings), returns_pkl_path)
    elif FLAGS.run_stage == "run_relaxation":
        relaxed_pdb_str, timings = run_relaxation(**argument_dict)
        save_object_as_pickle((relaxed_pdb_str, timings), returns_pkl_path)
    else:
        raise ValueError(f"No such stage: {FLAGS.run_stage}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["argument_path", "run_stage"])
    app.run(main)
