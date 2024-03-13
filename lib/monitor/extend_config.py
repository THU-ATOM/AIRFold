import os
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf
import copy

from lib.constant import CONFIG_ROOT


# path relative to the parent of the caller
rel_path = os.path.relpath(CONFIG_ROOT, Path(__file__).parent)
initialize(config_path=rel_path)


def get_conf_name(request_dict: dict) -> str:
    RUN_CONFIG = "run_config"
    NAME = "name"
    if RUN_CONFIG not in request_dict or NAME not in request_dict[RUN_CONFIG]:
        return RUN_CONFIG
    return request_dict[RUN_CONFIG][NAME]


def parse_default_overrides(run_config: dict) -> list:
    override_defaults = []
    for k, v in run_config.items():
        if isinstance(v, dict):
            for _k, _v in v.items():
                if isinstance(_v, dict):
                    override_defaults.append(f"{k}={_k}")
        else:
            override_defaults.append(f"{k}={v}")
    return override_defaults


def parse_value_overrides(run_config: dict) -> list:
    overrides = []
    for k, v in run_config.items():
        if isinstance(v, dict):
            _res = parse_value_overrides(v)
            overrides.extend([f"{k}.{o}" for o in _res])
        else:
            if isinstance(v, str):
                v = f'"{v}"'
            overrides.append(f"{k}={v}")
    return overrides


def parse_overrides(request_dict: dict) -> list:
    RUN_CONFIG = "run_config"
    if RUN_CONFIG not in request_dict:
        return []
    overrides = parse_default_overrides(request_dict[RUN_CONFIG])
    v_overrides = parse_value_overrides(request_dict[RUN_CONFIG])
    for i in v_overrides:
        if i not in overrides:
            overrides.append(i)
    return overrides


def extend_run_config(request_dict: dict) -> dict:
    RUN_CONFIG = "run_config"
    ret_request_dict = copy.copy(request_dict)
    if RUN_CONFIG in ret_request_dict:
        return ret_request_dict
    conf_name = get_conf_name(request_dict=ret_request_dict)
    cfg = compose(config_name=conf_name)
    ret_request_dict[RUN_CONFIG] = OmegaConf.to_container(cfg, resolve=True)
    return ret_request_dict


def generate_default_config(conf_name="run_config") -> dict:
    cfg = compose(config_name=conf_name)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return cfg


if __name__ == "__main__":
    partial_req = {"run_config": {"msa_select": "idle"}}
    try:
        cfg = extend_run_config(partial_req)
    except Exception as e:
        print(str(e))

    # cfg = compose(config_name="runConfig")
    print(cfg)
    print(OmegaConf.to_yaml(cfg))
