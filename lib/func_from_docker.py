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

"""Docker launch script for Alphafold docker image."""

from copy import deepcopy
import os
import getpass
import signal
import pickle as pkl
from time import sleep
from typing import Tuple
from pathlib import Path

import docker
from docker import types
from loguru import logger
from typing import Sequence, List, Dict, Union
from lib.tool import tool_utils as utils
from lib.constant import (
    AF_PARAMS_ROOT,
    PDB70_ROOT,
    PDBMMCIF_ROOT,
    TMP_ROOT,
)
import lib.datatypes as datatypes


_ROOT_MOUNT_DIRECTORY = "/"


def read_object_from_pickle(path: Union[str, Path]):
    with open(path, "rb") as fd:
        obj = pkl.load(file=fd)
    return obj


def _create_mount(mount_name: str, path: str) -> Tuple[types.Mount, str]:
    """Create a mount point for each file and directory used by the model."""
    path = Path(path).absolute()
    target_path = Path(_ROOT_MOUNT_DIRECTORY, mount_name)

    if path.is_dir():
        source_path = path
        mounted_path = target_path
    else:
        source_path = path.parent
        mounted_path = Path(target_path, path.name)
    if not source_path.exists():
        raise ValueError(
            f'Failed to find source directory "{source_path}" to '
            "mount in Docker container."
        )
    logger.info(f"Mounting {source_path} -> {target_path}")
    mount = types.Mount(target=str(target_path), source=str(source_path), type="bind")
    return mount, str(mounted_path)


def _run_operations_in_docker(
    argument_dict: dict,
    config: dict,
    stage: str,
    tmpdir: str,
    use_gpu: bool = True,
    gpu_devices: str = "",
    docker_image_name: str = "test/af2_run_stage",
    retry=5,
):
    """
    Run the operations in docker.
    tmpdir: the directory to store the temporary files, must be inside /tmp/... .
    """
    argument_dict = deepcopy(argument_dict)
    for k, v in config.items():
        if k not in argument_dict:
            argument_dict[k] = v

    logger.info(
        f"argument dict to _run_operations_in_docker stage {stage}: {[k for k, v in argument_dict.items()]}"
    )

    docker_user = f"{os.geteuid()}:{os.getegid()}"
    mounts = []
    mount, target_path = _create_mount("/data/protein", "/data/protein")
    mounts.append(mount)
    mount, _ = _create_mount("/tmp", "/tmp")
    mounts.append(mount)
    # mount, _ = _create_mount("/home/casp15", "/home/casp15")
    # mounts.append(mount)
    # logger.info(f"Mount info: {mounts}")

    client = docker.from_env()
    device_requests = (
        [docker.types.DeviceRequest(driver="nvidia", capabilities=[["gpu"]])]
        if use_gpu
        else None
    )
    gpu_devices = gpu_devices if use_gpu else ""
    # tmpdir = tempfile.mkdtemp(dir=TMP_ROOT)
    argument_path = os.path.join(tmpdir, "argument.pkl")
    logger.info(f"creating temporary file: {argument_path}")
    with open(argument_path, "wb") as fd:
        pkl.dump(argument_dict, file=fd, protocol=4)

    command_args = [
        f"--run_stage={stage}",
        f"--argument_path={argument_path}",
        f"--tmpdir={tmpdir}",
    ]
    logger.info(f"start to initiate container, using gpu devices: {gpu_devices}")
    
    # add params for timeout ctrl
    # - DOCKER_CLIENT_TIMEOUT=${DOCKER_CLIENT_TIMEOUT}
    # - COMPOSE_HTTP_TIMEOUT=${COMPOSE_HTTP_TIMEOUT}
    environment = {
        "NVIDIA_VISIBLE_DEVICES": gpu_devices,
        # The following flags allow us to make predictions on proteins that
        # would typically be too long to fit into GPU memory.
        "TF_FORCE_UNIFIED_MEMORY": "1",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0",
        "USERNAME": getpass.getuser(),
        "USERID": os.getuid(),
        "AF_PATH": "/home/casp15/code/alphafold",
        "DOCKER_CLIENT_TIMEOUT": "240",
        "COMPOSE_HTTP_TIMEOUT": "240",
    }

    logs = []
    for _retry in range(retry):
        container = client.containers.run(
            image=docker_image_name,
            command=command_args,
            device_requests=device_requests,
            remove=True,
            detach=True,
            mounts=mounts,
            # user=docker_user,
            stderr=True,
            environment=environment,
        )
        logger.info(f"initiate container complete")
        # Add signal handler to ensure CTRL+C also stops the running container.
        signal.signal(signal.SIGINT, lambda unused_sig, unused_frame: container.kill())
        logs = []
        for line in container.logs(stream=True, stderr=True, stdout=True):
            logs.append(line.strip().decode("utf-8"))
            logs = logs[-200:]
            print(line.strip().decode("utf-8"))
            # logger.info(line.strip().decode("utf-8"))
        logger.info(f"exit container")
        return_path = os.path.join(tmpdir, "returns.pkl")
        if not Path(return_path).exists():
            logger.error(f"return pickle {return_path} does not exist")
            if _retry == retry - 1:
                environment.update(
                    {
                        "http_proxy": "http://10.0.0.12:8001",
                        "https_proxy": "http://10.0.0.12:8001",
                    }
                )
                logger.info(f"last resort on setting proxy")
            sleep(60)
            continue
        else:
            return
    logs = "\n".join(logs)
    raise ValueError(
        f"_run_operations_in_docker failed on stage {stage}, latest logs are:\n{logs}"
    )


def search_template(
    input_sequence: str,
    template_searching_msa_path: str,
    pdb70_database_path: str = str(PDB70_ROOT / "pdb70"),
    hhsearch_binary_path: str = "hhsearch",
) -> Sequence[Dict]:
    argument_dict = {
        "input_sequence": input_sequence,
        "template_searching_msa_path": template_searching_msa_path,
        "pdb70_database_path": pdb70_database_path,
        "hhsearch_binary_path": hhsearch_binary_path,
    }
    with utils.tmpdir_manager(TMP_ROOT) as tmpdir:
        _run_operations_in_docker(
            argument_dict=argument_dict,
            config={},
            stage="search_template",
            tmpdir=tmpdir,
            use_gpu=False,
        )
        pdb_template_hits = read_object_from_pickle(os.path.join(tmpdir, "returns.pkl"))
    return pdb_template_hits


def make_template_feature(
    input_sequence: str,
    pdb_template_hits: Sequence[Dict],
    max_template_hits: int = 20,
    template_mmcif_dir: str = str(PDBMMCIF_ROOT / "mmcif_files"),
    max_template_date: str = "2022-05-31",
    obsolete_pdbs_path: str = str(PDBMMCIF_ROOT / "obsolete.dat"),
    kalign_binary_path: str = "kalign",
) -> Dict:
    argument_dict = {
        "input_sequence": input_sequence,
        "pdb_template_hits": pdb_template_hits,
        "max_template_hits": max_template_hits,
        "template_mmcif_dir": template_mmcif_dir,
        "max_template_date": max_template_date,
        "obsolete_pdbs_path": obsolete_pdbs_path,
        "kalign_binary_path": kalign_binary_path,
    }
    with utils.tmpdir_manager(TMP_ROOT) as tmpdir:
        _run_operations_in_docker(
            argument_dict=argument_dict,
            config={},
            stage="make_template_feature",
            tmpdir=tmpdir,
            use_gpu=False,
        )
        template_feature_dict = read_object_from_pickle(
            os.path.join(tmpdir, "returns.pkl")
        )
    return template_feature_dict


def monomer_msa2feature(
    sequence: str,
    target_name: str,
    msa_paths: List[str],
    template_feature: Dict,
    af2_config: Dict,
    model_name: str = "model_1",
    random_seed: int = 0,
) -> Tuple[datatypes.FeatureDict, Dict]:
    argument_dict = {
        "sequence": sequence,
        "target_name": target_name,
        "msa_paths": msa_paths,
        "template_feature": template_feature,
        "model_name": model_name,
        "random_seed": random_seed,
    }
    with utils.tmpdir_manager(TMP_ROOT) as tmpdir:
        _run_operations_in_docker(
            argument_dict=argument_dict,
            config=af2_config,
            stage="monomer_msa2feature",
            tmpdir=tmpdir,
            use_gpu=False,
        )
        feat, timings = read_object_from_pickle(os.path.join(tmpdir, "returns.pkl"))
    return feat


def predict_structure(
    af2_config: Dict,
    target_name: str,
    processed_feature: datatypes.FeatureDict,
    model_name: str = "model_1",
    data_dir: str = str(AF_PARAMS_ROOT),
    random_seed: int = 0,
    return_representations: bool = True,
    gpu_devices: str = "0",
) -> Tuple[Dict, str, dict]:
    argument_dict = {
        "target_name": target_name,
        "processed_feature": processed_feature,
        "model_name": model_name,
        "data_dir": data_dir,
        "random_seed": random_seed,
        "return_representations": return_representations,
    }
    with utils.tmpdir_manager(TMP_ROOT) as tmpdir:
        _run_operations_in_docker(
            argument_dict=argument_dict,
            config=af2_config,
            stage="predict_structure",
            tmpdir=tmpdir,
            use_gpu=True,
            gpu_devices=gpu_devices,
        )
        prediction_result, unrelaxed_pdb_str, timings = read_object_from_pickle(
            os.path.join(tmpdir, "returns.pkl")
        )
    return prediction_result, unrelaxed_pdb_str


def run_relaxation(
    unrelaxed_pdb_str: str, gpu_devices: str = "0"
) -> Tuple[datatypes.Protein, Dict]:
    argument_dict = {"unrelaxed_pdb_str": unrelaxed_pdb_str}

    with utils.tmpdir_manager(TMP_ROOT) as tmpdir:
        _run_operations_in_docker(
            argument_dict=argument_dict,
            config={},
            stage="run_relaxation",
            tmpdir=tmpdir,
            use_gpu=True,
            gpu_devices=gpu_devices,
        )
        relaxed_pdb_str, timings = read_object_from_pickle(
            os.path.join(tmpdir, "returns.pkl")
        )
    return relaxed_pdb_str
