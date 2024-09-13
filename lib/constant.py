"""This module defines CAMEO specific constants and tool classes.

"""
import sys
from pathlib import Path

from lib.utils.execute import rlaunch_exists


# Server
REQUEST_SERVER = "http://127.0.0.1"
BENCHMARK_SERVER = "http://127.0.0.1"

if rlaunch_exists():
    # Datasets
    SENDER_DATA_ROOT = Path("/data/protein/SENDER/")
    CAMEO_DATA_ROOT = Path("/sharefs/thuhc-data/CAMEO/data/")
    CASP15_DATA_ROOT = Path("/sharefs/thuhc-data/CASP15/data/")
    UNICLUST_ROOT = Path("/sharefs/thuhc-data/data/uniref")
    BFD_ROOT = Path("/sharefs/thuhc-data/data/bfd")
    AF_MGNIFY_ROOT = Path("/sharefs/thuhc-data/alphafold/mgnify")
    AF_UNIREF90_ROOT = Path("/sharefs/thuhc-data/alphafold/uniref90")
    AF_PARAMS_ROOT = Path("/sharefs/thuhc-data/alphafold/")
    UNICLUST_ESM_EMBEDDING = Path("/sharefs/thuhc-data/uniclust30/ESM_embeddings.pkl")
    BLAST_ROOT = Path("/sharefs/thuhc-data/blast_dbs/nr/nr")

    # Runtime
    TMP_ROOT = Path("/sharefs/thuhc-data/CAMEO/tmp/")
    # DB_PATH = Path("/sharefs/thuhc-data/CAMEO/database/cameo.db")
    LOG_ROOT = Path("/sharefs/thuhc-data/CAMEO/log/")
    TORCH_ROOT = Path("/sharefs/thuhc-data/torch_model/")
else:
    # Datasets
    SENDER_DATA_ROOT = Path("/data/protein/SENDER/")
    CAMEO_DATA_ROOT = Path("/data/protein/CAMEO/data/")
    CASP15_DATA_ROOT = Path("/data/protein/CASP15/data/")
    AF_PARAMS_ROOT = Path("/data/protein/alphafold/")
    UNICLUST_ESM_EMBEDDING = Path("/data/protein/uniclust30/ESM_embeddings.pkl")
    BFD_ROOT = Path("/data/protein/alphafold/bfd")
    BLAST_ROOT = Path("/data/protein/datasets_2022/blast_dbs/nr/nr")

    # upgrade: 2022-08-07 15:09:57
    PDB70_ROOT = Path("/data/protein/datasets_2022/pdb70")
    # PDB70_ROOT = Path("/data/protein/alphafold/pdb70")
    PDBMMCIF_ROOT = Path("/data/protein/datasets_2022/pdb_mmcif")
    UNICLUST_ROOT = Path("/data/protein/datasets_2022/uniclust30")
    AF_UNIREF90_ROOT = Path("/data/protein/datasets_2022/uniref90")
    AF_MGNIFY_ROOT = Path("/data/protein/datasets_2022/mgnify")

    # Runtime
    TMP_ROOT = Path("/data/protein/CAMEO/tmp/")
    # DB_PATH = Path("/data/protein/CAMEO/database/cameo.db")
    LOG_ROOT = Path("/data/protein/CAMEO/log/")
    SBFD_ROOT = Path("/data/protein/alphafold/small_bfd")
    TORCH_ROOT = Path("/data/protein/torch_model/")


# Config
CONFIG_ROOT = Path(__file__).parent.resolve() / "conf"

# Tools
COLABFOLD_PYTHON_PATH = Path("/usr/local/envs/colabfold/bin/python")
HHFILTER_PATH = Path("/usr/local/bin/hhfilter")
HHSEARCH_PATH = Path("/usr/local/bin/hhsearch")
LDDT_EXECUTE = Path(__file__).parent.resolve() / "tool" / "lddt-linux" / "lddt"
# TMALIGN_EXECUTE = Path("/usr/local/bin/TMalign")
# KALIGN_EXECUTE = Path("/usr/local/bin/kalign")
TMALIGN_EXECUTE = Path("TMalign")
KALIGN_EXECUTE = Path("kalign")
PROBCONS_EXECUTE = Path("/usr/local/bin/probcons")
CLUSTALO_EXECUTE = Path("/usr/local/bin/clustalo")


MONITOR_LOGGING_CONFIG = {
    "handlers": [
        {"sink": sys.stdout, "colorize": True},
        {
            "sink": LOG_ROOT / "monitor.log",
            "enqueue": True,
            "rotation": "1 month",
            "compression": "tar.gz",
        },
    ]
}

API_LOGGING_CONFIG = {
    "handlers": [
        {"sink": sys.stdout},
        {
            "sink": LOG_ROOT / "api.log",
            "enqueue": True,
            "rotation": "1 month",
            "compression": "tar.gz",
        },
    ]
}

FEISHU_WEBHOOK = (
    "https://open.feishu.cn/open-apis/bot/v2/hook/35818ba6-0b53-4717-83fb-7be4952a4f2d"
)
